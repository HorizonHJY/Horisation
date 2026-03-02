"""
market_controller.py
Flask Blueprint for the second-hand marketplace feature.
All routes require login. Images are uploaded to Cloudflare R2.
Listing metadata is stored in SQLite via market_db.py.
"""

from flask import Blueprint, request, jsonify

from Backend.Controller.auth_controller import login_required
from Backend.Controller import market_db, r2_manager

market_bp = Blueprint('market', __name__, url_prefix='/api/market')

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MAX_IMAGE_SIZE     = 5 * 1024 * 1024   # 5 MB
MAX_IMAGES         = 3
VALID_CATEGORIES   = {'electronics', 'clothing', 'books', 'furniture', 'other'}


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
    contact     = request.form.get('contact', '').strip()
    category    = request.form.get('category', 'other').strip()

    try:
        price = float(request.form.get('price', ''))
    except (ValueError, TypeError):
        return jsonify({'ok': False, 'error': 'Price must be a number.'}), 400

    if not title:
        return jsonify({'ok': False, 'error': 'Title is required.'}), 400
    if not description:
        return jsonify({'ok': False, 'error': 'Description is required.'}), 400
    if not contact:
        return jsonify({'ok': False, 'error': 'Contact info is required.'}), 400
    if category not in VALID_CATEGORIES:
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
    listing_id = market_db.create_listing(seller, title, description, price, category, contact)

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
    if 'contact'     in data: fields['contact']     = str(data['contact']).strip()
    if 'category'    in data:
        if data['category'] not in VALID_CATEGORIES:
            return jsonify({'ok': False, 'error': 'Invalid category.'}), 400
        fields['category'] = data['category']
    if 'price' in data:
        try:
            fields['price'] = float(data['price'])
        except (ValueError, TypeError):
            return jsonify({'ok': False, 'error': 'Price must be a number.'}), 400

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


# ── My listings ───────────────────────────────────────────────────────────────

@market_bp.route('/my', methods=['GET'])
@login_required
def my_listings():
    seller   = request.current_user['username']
    listings = market_db.get_my_listings(seller)
    return jsonify({'ok': True, 'listings': listings})
