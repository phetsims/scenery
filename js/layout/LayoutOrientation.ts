// Copyright 2022-2026, University of Colorado Boulder

/**
 * String union enumeration for orientation
 *
 * @author Jonathan Olson (PhET Interactive Simulations)
 */

export const LayoutOrientationValues = [ 'horizontal', 'vertical' ] as const;
export type LayoutOrientation = typeof LayoutOrientationValues[number];