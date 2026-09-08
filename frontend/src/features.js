/**
 * Feature flags — controls which roles can see each feature.
 * Role hierarchy: horizon > admin > svip > vip > user
 */

const VIP_AND_ABOVE = ['horizon', 'admin', 'svip', 'vip']

export const FEATURES = {
  darkMode:      ['horizon'],
  tarot:         ['horizon'],
  onlineGomoku:  ['horizon', 'admin', 'svip'],
  travelPlanner: VIP_AND_ABOVE,
  billSplit:     VIP_AND_ABOVE,
}

export function canAccess(userRole, feature) {
  return (FEATURES[feature] ?? []).includes(userRole)
}
