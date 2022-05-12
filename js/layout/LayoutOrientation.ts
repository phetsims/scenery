// Copyright 2022, University of Colorado Boulder

/**
 * String union enumeration for orientation
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export const LayoutOrientationValues = [ 'horizontal', 'vertical' ] as const;
export type LayoutOrientation = typeof LayoutOrientationValues[number];
