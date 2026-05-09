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
    data   = request.get_json() or {}
    slug   = data.get('slug', '').strip()
    label  = data.get('label', '').strip()
    order  = int(data.get('order', 0))
    active = bool(data.get('active', True))
    icon   = data.get('icon', 'fa-tag').strip() or 'fa-tag'
    if not slug or not label:
        return jsonify({'ok': False, 'error': 'slug and label are required.'}), 400
    cat = market_db.upsert_category(slug, label, order, active, icon)
    return jsonify({'ok': True, 'category': cat}), 201


@market_bp.route('/categories/<slug>', methods=['PUT'])
@admin_required
def update_category(slug):
    data   = request.get_json() or {}
    label  = data.get('label', '').strip()
    order  = int(data.get('order', 0))
    active = bool(data.get('active', True))
    icon   = data.get('icon', 'fa-tag').strip() or 'fa-tag'
    if not label:
        return jsonify({'ok': False, 'error': 'label is required.'}), 400
    cat = market_db.upsert_category(slug, label, order, active, icon)
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
    listings = market_db.get_all_listings(status='active')
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
    listing = market_db.get_listing(listing_id)
    if not listing:
        return jsonify({'ok': False, 'error': 'Listing not found.'}), 404
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
def user_listin