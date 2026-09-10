/**
 * Feature flags — controls which roles can see each feature.
 * Role hierarchy: horizon > admin > svip > vip > user
 */

const VIP_AND_ABOVE = ['horizon', 'admin', 'svip', 'vip']

/* Only list a feature here if it is actually restricted. Tarot was
   `['horizon']` while it was being built and is now open to everyone, so its
   entry is gone rather than expanded to name all five roles — a flag that
   lists every role is a gate that does nothing, and the next person has to
   read it to find that out. Open features simply carry no flag: see Market,
   Tasks, Message Board, Friends, Groups. */
export const FEATURES = {
  darkMode:      ['horizon'],
  onlineGomoku:  ['horizon', 'admin', 'svip'],
  travelPlanner: VIP_AND_ABOVE,
  billSplit:     VIP_AND_ABOVE,
}

export function canAccess(userRole, feature) {
  return (FEATURES[feature] ?? []).includes(userRole)
}
