"""
market_controller.py
Flask Blueprint for the second-hand marketplace feature.
All routes require login. Images are uploaded to Cloudflare R2.
Listing metadata is stored in SQLite via market_db.py.
"""

from flask import Blueprint, request, jsonify

from Backend.Controller.auth_controller import login_required, login_optional, admin_required
from Backend.Controller import market_db, r2_manager
from Backend.Controller.user_manager import user_manager

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
@login_optional
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
@login_optional
def list_listings():
    """Browse listings, paged newest-first.

    Query:
      limit  — rows per page, 1..200 (default 100)
      offset — rows to skip (default 0)

    Reserved listings owned by the signed-in buyer are appended only on the
    first page so they stay visible for receipt-confirm without multiplying
    across pages.
    """
    me = request.current_user['username'] if request.current_user else None
    try:
        limit = int(request.args.get('limit', 100))
    except (TypeError, ValueError):
        limit = 100
    try:
        offset = int(request.args.get('offset', 0))
    except (TypeError, ValueError):
        offset = 0
    limit  = max(1, min(limit, 200))
    offset = max(0, offset)

    listings = market_db.get_all_listings(status='active',
                                          limit=limit + 1, offset=offset) or []
    has_more = len(listings) > limit
    listings = listings[:limit]

    if me and offset == 0:
        extra = market_db.get_reserved_listings_for_buyer(me)
        if extra:
            seen = {l['id'] for l in listings}
            listings = listings + [l for l in extra if l['id'] not in seen]

    total = market_db.count_listings(status='active')
    return jsonify({'ok': True, 'listings': listings,
                    'has_more': has_more, 'offset': offset,
                    'limit': limit, 'total': total})


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

    # New shape: two boolean fields chosen independently. Legacy clients
    # that still send delivery_type keep working because the server derives
    # whichever side the caller didn't send.
    _truthy = {'1', 'true', 'True', 'on', 'yes'}
    raw_p = request.form.get('pickup_available')
    raw_d = request.form.get('delivery_available')
    pickup_available   = (raw_p in _truthy) if raw_p is not None else None
    delivery_available = (raw_d in _truthy) if raw_d is not None else None
    if pickup_available is None and delivery_available is None:
        delivery_type = request.form.get('delivery_type', 'pickup').strip()
        if delivery_type not in ('pickup', 'delivery', 'both'):
            delivery_type = 'pickup'
        pickup_available   = delivery_type in ('pickup', 'both')
        delivery_available = delivery_type in ('delivery', 'both')
    if not pickup_available and not delivery_available:
        pickup_available = True
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
        pickup_available=pickup_available,
        delivery_available=delivery_available,
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
@login_optional
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
    if 'pickup_available' in data:
        fields['pickup_available'] = bool(data['pickup_available'])
    if 'delivery_available' in data:
        fields['delivery_available'] = bool(data['delivery_available'])
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
    me       = request.current_user['username']
    is_admin = user_manager.check_permission(me, 'admin')
    r2_keys  = market_db.delete_listing(listing_id, me, allow_any=is_admin)

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


# ── Interest + Reserved (merged Message / I-want-this flow) ─────────────────
# Reach-out and I-want-this were two buttons for the same intent. They are
# now one: /interest posts a chat system bubble (seller sees the title and
# price inline) and the Market card drops the buyer back into the chat.
# Reserved is a seller-controlled flag independent of any buyer action.

@market_bp.route('/listings/<listing_id>/interest', methods=['POST'])
@login_required
def send_interest_route(listing_id):
    """Buyer expresses interest in a listing by posting a system bubble in
    the chat with the seller. The listing's state does not change.
    """
    me      = request.current_user['username']
    listing = market_db.get_listing(listing_id)
    if not listing or listing['status'] == 'sold':
        return jsonify({'ok': False, 'error': 'Listing is not available.'}), 400
    if listing['seller_username'] == me:
        return jsonify({'ok': False, 'error': 'That is your own listing.'}), 400

    import json as _json
    payload = {
        'listing_id': listing['id'],
        'title': listing['title'],
        'price': listing.get('price'),
        'status': 'pending',
    }
    content = f"[sys:interest]{_json.dumps(payload, ensure_ascii=False)}"
    ua, ub   = market_db._friend_pair(me, listing['seller_username'])
    room_key = f'{ua}:{ub}'
    msg      = market_db.save_chat_message(room_key, me, content)

    from .socketio_instance import socketio
    socketio.emit('chat_message', msg, room=f"user_{me}")
    socketio.emit('chat_message', msg, room=f"user_{listing['seller_username']}")
    return jsonify({'ok': True, 'message': msg,
                    'seller_username': listing['seller_username']}), 201


@market_bp.route('/listings/<listing_id>/reserve', methods=['PUT'])
@login_required
def reserve_listing_route(listing_id):
    """Seller marks a listing reserved so other buyers see it is held."""
    me      = request.current_user['username']
    listing = market_db.get_listing(listing_id)
    if not listing:
        return jsonify({'ok': False, 'error': 'Listing not found.'}), 404
    if listing['seller_username'] != me:
        return jsonify({'ok': False, 'error': 'Only the seller can reserve.'}), 403
    if listing['status'] not in ('active', 'reserved'):
        return jsonify({'ok': False, 'error': 'Only active listings can be reserved.'}), 400
    updated = market_db.set_listing_status(listing_id, 'reserved')
    return jsonify({'ok': True, 'listing': updated})


@market_bp.route('/listings/<listing_id>/unreserve', methods=['PUT'])
@login_required
def unreserve_listing_route(listing_id):
    """Seller releases a reserved listing back to active."""
    me      = request.current_user['username']
    listing = market_db.get_listing(listing_id)
    if not listing:
        return jsonify({'ok': False, 'error': 'Listing not found.'}), 404
    if listing['seller_username'] != me:
        return jsonify({'ok': False, 'error': 'Only the seller can unreserve.'}), 403
    if listing['status'] != 'reserved':
        return jsonify({'ok': False, 'error': 'Listing is not reserved.'}), 400
    updated = market_db.set_listing_status(listing_id, 'active')
    return jsonify({'ok': True, 'listing': updated})


# ── Wanted (求购) ────────────────────────────────────────────────────────────
# Reverse of a listing: the buyer posts what they're looking for, sellers see
# it in Market's Wanted tab and DM the buyer directly (not through intents —
# a wanted post has no listing to reserve, and the two sides' roles are
# swapped). When the buyer finds what they wanted, they mark it fulfilled.

def _parse_price(raw):
    raw = (raw or '').strip() if isinstance(raw, str) else raw
    if raw in (None, ''):
        return None
    try:
        v = float(raw)
        return v if v >= 0 else None
    except (ValueError, TypeError):
        return None


def _parse_wanted_body(body: dict) -> tuple[dict, str | None]:
    """Return (clean_fields, error_msg). error_msg is None on success."""
    title       = (body.get('title') or '').strip()
    description = (body.get('description') or '').strip()
    category    = (body.get('category') or 'other').strip()
    contact     = (body.get('contact') or '').strip()
    condition   = (body.get('condition') or 'any').strip()

    if not title:
        return {}, 'Title is required.'
    if not description:
        return {}, 'Description is required.'
    if not market_db.category_slug_valid(category):
        return {}, 'Invalid category.'
    if condition not in ('any', 'new', 'used'):
        condition = 'any'

    price_min = _parse_price(body.get('price_min'))
    price_max = _parse_price(body.get('price_max'))
    if price_min is not None and price_max is not None and price_min > price_max:
        return {}, 'Minimum price cannot exceed maximum.'

    return {
        'title': title[:100], 'description': description, 'category': category,
        'contact': contact[:200], 'price_min': price_min, 'price_max': price_max,
        'condition': condition,
    }, None


@market_bp.route('/wanted', methods=['GET'])
@login_optional
def list_wanted():
    """The Wanted feed — open posts, newest first. Optional ?category= filter.
    Open to guests like /listings, so the Market tab works without login."""
    category = request.args.get('category') or None
    return jsonify({'ok': True, 'items': market_db.list_wanted(category)})


@market_bp.route('/wanted', methods=['POST'])
@login_required
def create_wanted():
    """Buyer posts a wanted item."""
    body = request.get_json(silent=True) or {}
    fields, err = _parse_wanted_body(body)
    if err:
        return jsonify({'ok': False, 'error': err}), 400
    me = request.current_user['username']
    wid = market_db.create_wanted(buyer=me, **fields)
    return jsonify({'ok': True, 'item': market_db.get_wanted(wid)}), 201


@market_bp.route('/wanted/mine', methods=['GET'])
@login_required
def my_wanted():
    """Everything the caller has ever posted (open / fulfilled / removed)."""
    me = request.current_user['username']
    return jsonify({'ok': True, 'items': market_db.list_my_wanted(me)})


@market_bp.route('/wanted/<wanted_id>', methods=['GET'])
@login_optional
def get_wanted(wanted_id):
    item = market_db.get_wanted(wanted_id)
    if not item:
        return jsonify({'ok': False, 'error': 'Wanted post not found.'}), 404
    return jsonify({'ok': True, 'item': item})


@market_bp.route('/wanted/<wanted_id>', methods=['PUT'])
@login_required
def update_wanted(wanted_id):
    body = request.get_json(silent=True) or {}
    fields, err = _parse_wanted_body(body)
    if err:
        return jsonify({'ok': False, 'error': err}), 400
    me = request.current_user['username']
    if not market_db.update_wanted(wanted_id, me, **fields):
        return jsonify({'ok': False, 'error': 'Not yours, or no longer exists.'}), 404
    return jsonify({'ok': True, 'item': market_db.get_wanted(wanted_id)})


@market_bp.route('/wanted/<wanted_id>/fulfilled', methods=['POST'])
@login_required
def mark_wanted_fulfilled(wanted_id):
    """Buyer found what they wanted. Post falls out of the open feed."""
    me = request.current_user['username']
    if not market_db.fulfill_wanted(wanted_id, me):
        return jsonify({'ok': False, 'error': 'Not yours, or already fulfilled.'}), 400
    return jsonify({'ok': True, 'item': market_db.get_wanted(wanted_id)})


@market_bp.route('/wanted/<wanted_id>/reopen', methods=['POST'])
@login_required
def reopen_wanted_route(wanted_id):
    """Undo a fulfill — put the wanted post back in the open feed."""
    me = request.current_user['username']
    if not market_db.reopen_wanted(wanted_id, me):
        return jsonify({'ok': False, 'error': 'Not yours, or not fulfilled.'}), 400
    return jsonify({'ok': True, 'item': market_db.get_wanted(wanted_id)})


@market_bp.route('/wanted/<wanted_id>', methods=['DELETE'])
@login_required
def delete_wanted(wanted_id):
    me = request.current_user['username']
    if not market_db.delete_wanted(wanted_id, me):
        return jsonify({'ok': False, 'error': 'Not yours, or no longer exists.'}), 404
    return jsonify({'ok': True})
