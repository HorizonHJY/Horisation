# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Horisation is a Flask-based web application for CSV/Excel data analysis, financial modeling, and personal productivity. It provides file upload, preview, data summarization, financial calculation capabilities, user management, private notes, and memo management through a secure web interface with role-based access control.

## Architecture

### Backend Structure
- **app.py**: Main Flask application entry point with route definitions and user authentication
- **Backend/Controller/**: Request handling and business logic
  - `csvcontroller.py`: Blueprint-based API endpoints for CSV/Excel operations with encoding fallback
  - `csv_handling.py`: Core data processing functions (cleaning, merging, exporting)
  - `user_manager.py`: User authentication, session management, and role-based access control
  - `auth_controller.py`: Authentication API endpoints (login, logout, user management)
  - `notes_manager.py`: Private notes and diary management system
  - `notes_controller.py`: Private notes API endpoints
  - `memos_controller.py`: User-isolated memo/task management API endpoints
- **Backend/Horfunc/**: Financial and analytical functions
  - `finpkg.py`: Monte Carlo price simulation using Geometric Brownian Motion
- **Backend/Sandbox/**: Experimental/testing code

### Frontend Structure
- **Template/**: Jinja2 HTML templates
  - `Home.html`: Base layout template (login-protected)
  - `CSV.html`: CSV upload and preview interface
  - `hormemo.html`: User-isolated memo/task interface with real-time updates
  - `limit.html`: Limit tracking interface
  - `horbase.html`: Shared base template with user authentication context
  - `auth/`: Authentication templates
    - `login.html`: User login page with modern design
  - `notes/`: Private notes templates
    - `notes.html`: Private notes management with categories and tags
- **Static/**: Frontend assets
  - `js/horcsv.js`: CSV upload, drag-drop, preview/summary client logic
  - `js/hormemo.js`: Memo functionality (updated for user isolation)
  - `css/`: Styling files
  - `pic/`: Images

### Data Flow
1. User authentication: Login required for all protected routes
2. User uploads CSV/Excel via drag-drop or file picker
3. Frontend JavaScript (horcsv.js) sends file to API endpoint with session validation
4. Backend attempts UTF-8 encoding, falls back to GBK/GB2312/Big5/etc.
5. Pandas processes the file and returns preview or summary
6. Results rendered in browser table

### User Management System
**Authentication Flow**:
1. All routes require login (redirect to `/login` if not authenticated)
2. User credentials validated against `_data/users.json`
3. Session tokens stored in `_data/sessions.json` with 24-hour expiry
4. Role-based access control for different features

**Data Storage Structure**:
- `_data/users.json`: User accounts with plaintext passwords (development mode)
- `_data/sessions.json`: Active user sessions with expiry timestamps
- `_data/notes/{username}_notes.json`: User-isolated private notes
- User-specific memo data stored within user objects

## Development Commands

### Running the Application
```bash
# Start development server
python app.py

# Server runs on http://localhost:5000 with debug mode enabled
```

### Key Dependencies
```bash
# Install required packages
pip install flask pandas numpy openpyxl xlrd
```

**Note**: Excel support requires:
- `.xlsx` files → `openpyxl`
- `.xls` files → `xlrd`

### Testing Endpoints

**Preview API** (first N rows):
```bash
# Default UTF-8
curl -X POST -F "file=@data.csv" "http://localhost:5000/api/csv/preview?n=10"

# Custom encoding
curl -X POST -F "file=@data.csv" "http://localhost:5000/api/csv/preview?n=5&encoding=gbk"

# Custom separator
curl -X POST -F "file=@data.csv" "http://localhost:5000/api/csv/preview?sep=%3B"  # semicolon
```

**Summary API** (full file statistics):
```bash
curl -X POST -F "file=@data.csv" "http://localhost:5000/api/csv/summary"
```

## Important Technical Details

### Encoding Handling
The `csvcontroller.py` implements robust encoding detection with fallback chain:
1. UTF-8 (with PyArrow if available)
2. UTF-8-SIG (BOM handling)
3. Local encodings: GBK → GB2312 → Big5 → Shift_JIS → CP1252
4. Final fallback: Latin1

**When adding encoding support**: Update the fallback list in `_read_csv_with_fallback()` at line 58.

### File Upload Configuration
- Max file size: 100MB (`MAX_BYTES` in csvcontroller.py:11)
- Max request size: 20MB (`MAX_CONTENT_LENGTH` in app.py:16)
- Allowed extensions: `.csv`, `.xls`, `.xlsx`
- Upload directory: `_uploads/` (created automatically)

**To change limits**: Update both `MAX_BYTES` and `app.config['MAX_CONTENT_LENGTH']`.

### Data Processing Pipeline

**CSV Cleaning** (`csv_handling.py`):
- Column names: Uppercase + strip whitespace + deduplicate
- Cell values: Strip whitespace for string columns
- Deduplication: Drop duplicates with configurable `subset` and `keep` parameters

**DataFrame Merging** (`concat_dfs()`):
- Vertical concatenation (row-wise)
- Column union (missing columns filled with NaN)
- Optional column aliasing for name normalization

### Financial Functions

**Monte Carlo Simulation** (`finpkg.py`):
```python
simulate_price(S0, vol_annual, T, seed=None, basis=252)
```
- Uses Geometric Brownian Motion: `S_T = S0 * exp(-0.5σ²T + σ√T * Z)`
- Default basis: 252 trading days
- Reads from `PFE_Results.xlsx` for batch calculations

## Common Development Tasks

### Adding New API Endpoints
1. Add route function to `csvcontroller.py` blueprint
2. Use `_get_file_and_bytes()` helper for file validation
3. Return `jsonify({'ok': True/False, ...})` format
4. Register blueprint in `app.py` if creating new module

### Adding New Templates
1. Create HTML in `Template/` directory
2. Extend `Home.html` base template using `{% extends "Home.html" %}`
3. Add route in `app.py` with `active_page` parameter
4. Link static files via `{{ url_for('static', filename='...') }}`

### Modifying File Processing
- **Preview logic**: Edit `read_csv_preview()` in csvcontroller.py:127
- **Summary logic**: Edit `summarize_csv()` in csvcontroller.py:138
- **Type inference**: Regex patterns in summarize_csv:166-172
- **Missing value detection**: Line 147 checks for '', 'nan', 'None', NaN

### Frontend JavaScript Patterns
- Uses vanilla JavaScript with IIFE pattern `(() => { ... })()`
- DOM selection: `const $ = (id) => document.getElementById(id)`
- Drag-drop counter pattern prevents flicker (dragCounter)
- Fetch API with FormData for file uploads

## User Roles and Permissions

### Role Hierarchy
- **horizon** (Level 100): Super Administrator
  - Full system access including user management
  - Sectors: `['all']` (can access all features)
  - Permissions: `['admin', 'read', 'write', 'delete', 'user_manage']`

- **horizonadmin** (Level 90): Horizon Administrator
  - Administrative functions and user management
  - Sectors: `['horizon', 'admin']`
  - Permissions: `['admin', 'read', 'write', 'delete']`

- **vip1** (Level 80): VIP Tier 1 User
  - Advanced features and data access
  - Sectors: `['vip', 'general']`
  - Permissions: `['read', 'write']`

- **vip2** (Level 70): VIP Tier 2 User
  - Mid-tier features and data access
  - Sectors: `['vip', 'general']`
  - Permissions: `['read', 'write']`

- **vip3** (Level 60): VIP Tier 3 User
  - Basic VIP features and data access
  - Sectors: `['vip', 'general']`
  - Permissions: `['read', 'write']`

- **user** (Level 10): Standard User
  - Basic features and read-only access
  - Sectors: `['general']`
  - Permissions: `['read']`

### Development Credentials
- **Admin Account**: `horizon` / `horizon`
- **Test User**: `fanfan0315` / `yyf`

**Note**: Passwords are stored in plaintext for development convenience. In production, implement proper password hashing.

## Routes

### Public Routes
- `/login` → User login page

### Protected Routes (Login Required)
- `/` → Home dashboard
- `/csv` → CSV upload workspace
- `/hormemo` → User-isolated memo interface
- `/notes` → Private notes and diary
- `/limit` → Limit tracking
- `/profile` → User profile management

### Admin Routes (Admin Permission Required)
- `/admin/users` → User management interface

### API Endpoints

#### Authentication API (`/api/auth/`)
- `POST /login` → User authentication
- `POST /logout` → User logout (login required)
- `GET /check-session` → Validate current session
- `GET /profile` → Get current user info (login required)
- `POST /register` → Create new user (admin required)
- `GET /users` → List all users (admin required)
- `PUT /users/<username>/role` → Update user role (admin required)
- `PUT /users/<username>/status` → Activate/deactivate user (admin required)
- `POST /permissions/check` → Check user permissions (login required)

#### CSV Processing API (`/api/csv/`)
- `POST /preview` → Preview first N rows
- `POST /summary` → Full file statistics

#### Notes API (`/api/notes/`) - All require login
- `GET /` → List user's notes
- `POST /` → Create new note
- `GET /<note_id>` → Get specific note
- `PUT /<note_id>` → Update note
- `DELETE /<note_id>` → Delete note
- `GET /categories` → List note categories
- `GET /search` → Search notes
- `GET /stats` → Get notes statistics

#### Memos API (`/api/memos/`) - All require login
- `GET /` → List user's memos (with filtering)
- `POST /` → Create new memo
- `PUT /<memo_id>` → Update memo
- `DELETE /<memo_id>` → Delete memo
- `GET /stats` → Get memo statistics

## Template Variables
- `active_page`: Highlights current nav item ('home', 'csv', 'hormemo', 'limit')
- Static files: `{{ url_for('static', filename='path') }}`
- Templates: `{{ url_for('template', filename='path') }}`
