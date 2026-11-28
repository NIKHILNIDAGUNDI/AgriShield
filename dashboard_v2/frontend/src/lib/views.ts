export const views = ['overview', 'analytics', 'simulator'] as const
export type View = (typeof views)[number]
