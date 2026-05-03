/**
 * Returns an error string if the password fails requirements, or null if valid.
 * Requirements: 8+ chars, uppercase, lowercase, digit.
 */
export function validatePassword(pw) {
  if (pw.length < 8)        return 'Password must be at least 8 characters.'
  if (!/[A-Z]/.test(pw))    return 'Must contain at least one uppercase letter.'
  if (!/[a-z]/.test(pw))    return 'Must contain at least one lowercase letter.'
  if (!/\d/.test(pw))       return 'Must contain at least one number.'
  return null
}
