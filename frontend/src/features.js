/**
 * Feature flags — controls which roles can see each feature.
 * When a feature matures, add more roles to its array.
 *
 * Roles: 'horizon' | 'horizonadmin' | 'vip1' | 'vip2' | 'vip3' | 'user' | 'test'
 */
export const FEATURES = {
  darkMode:      ['horizon', 'test'],
  onlineGomoku:  ['horizon', 'horizonadmin', 'vip3'],
}

/** Returns true if the given role can access a feature. */
export function canAccess(userRole, feature) {
  return (FEATURES[feature] ?? []).includes(userRole)
}
