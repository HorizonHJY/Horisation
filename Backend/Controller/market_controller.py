"""
market_controller.py
Flask Blueprint for the second-hand marketplace feature.
All routes require login. Images are uploaded to Cloudflare R2.
Listing metadata is stored in SQLite via market_db.py.
"""

from flask import Blueprint, request, jsonify

from Backend.Controller.auth_controller import login_required, admin_required
from Backend.Controller import market_db, r2_manager

market_bp = Blueprint('market', __name__, url_prefix='/api/market')

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MAX_IMAGE_SIZE     = 5 * 1024 * 1024   # 5 MB
MAX_IMAGES         = 3


def _validate_image(file):
    """Return error string or None if file is valid."""
    import os
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return f"'{file.filename}' is not a JPEG or PNG."
    file.seek(0, 2)
    size = file.tell()
    file.seek(0)
    if size > MAX_IMAGE_SIZE:
        return f"'{file.filename}' exceeds 5 MB limit."
    return None


# ── Categories ───────────────────────────────────────────────────────────────

@market_bp.route('/categories', methods=['GET'])
@login_required
def list_categories():
    """Return active categories for create/edit forms."""
    return jsonify({'ok': True, 'categories': market_db.get_categories(active_only=True)})


@market_bp.route('/categories/all', methods=['GET'])
@admin_required
def list_all_categories():
    """Admin: return all categories including inactive ones."""
    return jsonify({'ok': True, 'categories': market_db.get_categories(active_only=False)})


@market_bp.route('/categories', methods=['POST'])
@admin_required
def create_category():
    data     = request.get_json() or {}
    slug     = data.get('slug', '').strip()
    label    = data.get('label', '').strip()
    label_zh = (data.get('label_zh') or '').strip()
    order    = int(data.get('order', 0))
    active   = bool(data.get('active', True))
    icon     = data.get('icon', 'fa-tag').strip() or 'fa-tag'
    if not slug or not label:
        return jsonify({'ok': False, 'error': 'slug and label are required.'}), 400
    cat = market_db.upsert_category(slug, label, order, active, icon, label_zh)
    return jsonify({'ok': True, 'category': cat}), 201


@market_bp.route('/categories/<slug>', methods=['PUT'])
@admin_required
def update_category(slug):
    data     = request.get_json() or {}
    label    = data.get('label', '').strip()
    label_zh = (data.get('label_zh') or '').strip()
    order    = int(data.get('order', 0))
    active   = bool(data.get('active', True))
    icon     = data.get('icon', 'fa-tag').strip() or 'fa-tag'
    if not label:
        return jsonify({'ok': False, 'error': 'label is required.'}), 400
    cat = market_db.upsert_category(slug, label, order, active, icon, label_zh)
    return jsonify({'ok': True, 'category': cat})


@market_bp.route('/categories/<slug>', methods=['DELETE'])
@admin_required
def delete_category_route(slug):
    ok = market_db.delete_category(slug)
    if not ok:
        return jsonify({'ok': False, 'error': 'Category not found.'}), 404
    return jsonify({'ok': True})


# ── Browse all active listings ────────────────────────────────────────────────

@market_bp.route('/listings', methods=['GET'])
@login_required
def list_listings():
    """Browse listings.

    Returns all active listings, plus any RESERVED listing that is currently
    being bought by the requesting user (so they can still see it and confirm
    receipt / complete the trade after the seller accepted).
    """
    me = request.current_user['username']
    listings = market_db.get_all_listings(status='active') or []
    # A reserved listing must stay visible to the buyer who owns the accepted intent.
    extra = market_db.get_reserved_listings_for_buyer(me)
    if extra:
        seen = {l['id'] for l in listings}
        listings = listings + [l for l in extra if l['id'] not in seen]
    return jsonify({'ok': True, 'listings': listings})


# ── Create listing ────────────────────────────────────────────────────────────

@market_bp.route('/listings', methods=['POST'])
@login_required
def create_listing():
    seller = request.current_user['username']

    title       = request.form.get('title', '').strip()
    description = request.form.get('description', '').strip()
    category    = request.form.get('category', 'other').strip()

    try:
        price = float(request.form.get('price', ''))
    except (ValueError, TypeError):
        return jsonify({'ok': False, 'error': 'Price must be a number.'}), 400

    original_price = None
    raw_op = request.form.get('original_price', '').strip()
    if raw_op:
        try:
            original_price = float(raw_op)
            if original_price < 0:
                original_price = None
        except (ValueError, TypeError):
            pass

    delivery_type = request.form.get('delivery_type', 'pickup').strip()
    if delivery_type not in ('pickup', 'delivery', 'both'):
        delivery_type = 'pickup'
    delivery_fee = None
    raw_df = request.form.get('delivery_fee', '').strip()
    if raw_df:
        try:
            delivery_fee = float(raw_df)
            if delivery_fee < 0:
                delivery_fee = None
        except (ValueError, TypeError):
            pass

    if not title:
        return jsonify({'ok': False, 'error': 'Title is required.'}), 400
    if not description:
        return jsonify({'ok': False, 'error': 'Description is required.'}), 400
    if not market_db.category_slug_valid(category):
        return jsonify({'ok': False, 'error': 'Invalid category.'}), 400
    if price < 0:
        return jsonify({'ok': False, 'error': 'Price cannot be negative.'}), 400

    files = request.files.getlist('images')
    if len(files) > MAX_IMAGES:
        return jsonify({'ok': False, 'error': f'Maximum {MAX_IMAGES} images allowed.'}), 400

    # Validate all images before uploading anything
    for f in files:
        if f and f.filename:
            err = _validate_image(f)
            if err:
                return jsonify({'ok': False, 'error': err}), 400

    # Create listing row
    listing_id = market_db.create_listing(
        seller, title, description, price, category, '',
        original_price=original_price,
        delivery_type=delivery_type,
        delivery_fee=delivery_fee,
    )

    # Upload images to R2
    uploaded_keys = []
    try:
        for order, f in enumerate(files):
            if f and f.filename:
                r2_key, r2_url = r2_manager.upload_image(f.stream, f.filename)
                market_db.add_image(listing_id, r2_url, r2_key, order)
                uploaded_keys.append(r2_key)
    except Exception as e:
        # Clean up uploaded images on failure
        for key in uploaded_keys:
            r2_manager.delete_image(key)
        market_db.delete_listing(listing_id, seller)
        return jsonify({'ok': False, 'error': f'Image upload failed: {str(e)}'}), 500

    listing = market_db.get_listing(listing_id)
    return jsonify({'ok': True, 'listing': listing}), 201


# ── Get single listing ────────────────────────────────────────────────────────

@market_bp.route('/listings/<listing_id>', methods=['GET'])
@login_required
def get_listing(listing_id):
    """Fetch one listing.

    `?track=0` returns fresh data without counting another view. The client
    sends it when this browser session has already been counted, so reopening
    a listing shows current price and status without inflating the counter.
    """
    listing = market_db.get_listing(listing_id)
    if not listing:
        return jsonify({'ok': False, 'error': 'Listing not found.'}), 404
    if request.args.get('track') != '0':
        market_db.increment_view_count(listing_id)
        listing['view_count'] += 1   # reflect the increment in this response
    return jsonify({'ok': True, 'listing': listing})


# ── Edit listing (seller only) ────────────────────────────────────────────────

@market_bp.route('/listings/<listing_id>', methods=['PUT'])
@login_required
def update_listing(listing_id):
    seller = request.current_user['username']
    data   = request.get_json() or {}

    fields = {}
    if 'title'       in data: fields['title']       = str(data['title']).strip()
    if 'description' in data: fields['description'] = str(data['description']).strip()
    if 'category'    in data:
        if not market_db.category_slug_valid(data['category']):
            return jsonify({'ok': False, 'error': 'Invalid category.'}), 400
        fields['category'] = data['category']
    if 'price' in data:
        try:
            fields['price'] = float(data['price'])
        except (ValueError, TypeError):
            return jsonify({'ok': False, 'error': 'Price must be a number.'}), 400
    if 'original_price' in data:
        raw = data['original_price']
        if raw == '' or raw is None:
            fields['original_price'] = None
        else:
            try:
                fields['original_price'] = float(raw)
            except (ValueError, TypeError):
                return jsonify({'ok': False, 'error': 'Original price must be a number.'}), 400

    if 'delivery_type' in data:
        dt = str(data['delivery_type']).strip()
        if dt in ('pickup', 'delivery', 'both'):
            fields['delivery_type'] = dt
    if 'delivery_fee' in data:
        raw = data['delivery_fee']
        if raw == '' or raw is None:
            fields['delivery_fee'] = None
        else:
            try:
                fields['delivery_fee'] = float(raw)
            except (ValueError, TypeError):
                return jsonify({'ok': False, 'error': 'Delivery fee must be a number.'}), 400

    ok = market_db.update_listing(listing_id, seller, **fields)
    if not ok:
        return jsonify({'ok': False, 'error': 'Listing not found or permission denied.'}), 404

    listing = market_db.get_listing(listing_id)
    return jsonify({'ok': True, 'listing': listing})


# ── Delete listing (seller only) ──────────────────────────────────────────────

@market_bp.route('/listings/<listing_id>', methods=['DELETE'])
@login_required
def delete_listing(listing_id):
    seller   = request.current_user['username']
    r2_keys  = market_db.delete_listing(listing_id, seller)

    if r2_keys is None:
        return jsonify({'ok': False, 'error': 'Listing not found or permission denied.'}), 404

    # Remove images from R2 (best-effort)
    for key in r2_keys:
        r2_manager.delete_image(key)

    return jsonify({'ok': True})


# ── Mark as sold (seller only) ────────────────────────────────────────────────

@market_bp.route('/listings/<listing_id>/sold', methods=['POST'])
@login_required
def mark_sold(listing_id):
    seller = request.current_user['username']
    ok     = market_db.mark_sold(listing_id, seller)
    if not ok:
        return jsonify({'ok': False, 'error': 'Listing not found or permission denied.'}), 404
    return jsonify({'ok': True})


# ── Restore sold listing (seller only) ───────────────────────────────────────

@market_bp.route('/listings/<listing_id>/restore', methods=['POST'])
@login_required
def restore_listing(listing_id):
    seller = request.current_user['username']
    ok     = market_db.restore_listing(listing_id, seller)
    if not ok:
        return jsonify({'ok': False, 'error': 'Listing not found, not sold, or permission denied.'}), 404
    listing = market_db.get_listing(listing_id)
    return jsonify({'ok': True, 'listing': listing})


# ── My listings ───────────────────────────────────────────────────────────────

@market_bp.route('/my', methods=['GET'])
@login_required
def my_listings():
    seller   = request.current_user['username']
    listings = market_db.get_my_listings(seller)
    return jsonify({'ok': True, 'listings': listings})


# ── Active listings for a specific user ───────────────────────────────────────

@market_bp.route('/user/<username>', methods=['GET'])
@login_required
def user_listings(username):
    listings = market_db.get_active_listings_by_user(username)
    return jsonify({'ok': True, 'listings': listings})


# ── Trade-intent flow (意向成单流) ────────────────────────────────────────────

@market_bp.route('/listings/<listing_id>/intent', methods=['POST'])
@login_required
def create_intent_route(listing_id):
    """Buyer expresses 'I want this'. Seller must own an active listing."""
    me      = request.current_user['username']
    body    = request.get_json() or {}
    message = (body.get('message') or '').strip() or None

    # Seller must own an active listing (look it up to get the real seller).
    listing = market_db.get_listing(listing_id)
    if not listing or listing['status'] != 'active':
        return jsonify({'ok': False, 'error': 'Listing not available for purchase.'}), 400
    if listing['seller_username'] == me:
        return jsonify({'ok': False, 'error': "You can't buy your own listing."}), 400
    if market_db.has_active_intent(listing_id, me):
        return jsonify({'ok': False, 'error': 'You already expressed interest. Wait for the seller.'}), 400

    intent = market_db.create_intent(listing_id, me, listing['seller_username'], message)
    if not intent:
        return jsonify({'ok': False, 'error': 'Unable to express interest right now.'}), 400

    from .friends_socket import notify_intent
    notify_intent('trade_intent', intent)
    return jsonify({'ok': True, 'intent': intent}), 201


@market_bp.route('/intents/outgoing', methods=['GET'])
@login_required
def my_intents():
    """Intents I placed as a buyer."""
    me = request.current_user['username']
    return jsonify({'ok': True, 'intents': market_db.get_my_buy_intents(me)})


@market_bp.route('/intents/incoming', methods=['GET'])
@login_required
def incoming_intents():
    """Intents buyers placed on my listings."""
    me = request.current_user['username']
    return jsonify({'ok': True, 'intents': market_db.get_seller_intents(me)})


@market_bp.route('/intents/<intent_id>/accept', methods=['PUT'])
@login_required
def accept_intent_route(intent_id):
    """Seller accepts a pending intent -> listing reserved, other intents declined."""
    me    = request.current_user['username']
    res   = market_db.accept_intent(intent_id, me)
    if not res:
        return jsonify({'ok': False, 'error': 'Intent not found, or listing/ownership changed.'}), 400
    from .friends_socket import notify_intent
    notify_intent('trade_intent_accepted', res)
    return jsonify({'ok': True, 'intent': res})


@market_bp.route('/intents/<intent_id>/decline', methods=['PUT'])
@login_required
def decline_intent_route(intent_id):
    """Seller declines a pending intent."""
    me  = request.current_user['username']
    ok  = market_db.decline_intent(intent_id, me)
    if not ok:
        return jsonify({'ok': False, 'error': 'Intent not found or not pending.'}), 400
    intent = market_db.get_intent(intent_id)
    from .friends_socket import notify_intent
    notify_intent('trade_intent_declined', intent)
    return jsonify({'ok': True, 'intent': intent})


@market_bp.route('/intents/<intent_id>/cancel', methods=['PUT'])
@login_required
def cancel_intent_route(intent_id):
    """Either party cancels before completion. Frees a reserved listing back to active."""
    me = request.current_user['username']
    if not market_db.cancel_intent(intent_id, me):
        return jsonify({'ok': False, 'error': 'Cannot cancel this intent.'}), 400
    intent = market_db.get_intent(intent_id)
    from .friends_socket import notify_intent
    notify_intent('trade_intent_cancelled', intent)
    return jsonify({'ok': True, 'intent': intent})


@market_bp.route('/intents/<intent_id>/complete', methods=['PUT'])
@login_required
def complete_intent_route(intent_id):
    """Buyer confirms receipt -> listing sold. Only the buyer of an accepted intent may call."""
    me   = request.current_user['username']
    res  = market_db.complete_intent(intent_id, me)
    if not res:
        return jsonify({'ok': False, 'error': 'Not found, not yours, or listing not reserved.'}), 400
    from .friends_socket import notify_intent
    notify_intent('trade_intent_completed', res)
    return jsonify({'ok': True, 'intent': res})
