// Copyright 2023, University of Colorado Boulder

/**
 * Fill rules, for determining how to fill a path (given the winding number of a face)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

type FillRule = 'nonzero' | 'evenodd' | 'positive' | 'negative';

export default FillRule;

export const isWindingIncluded = ( windingNumber: number, fillRule: FillRule ): boolean => {
  switch( fillRule ) {
    case 'nonzero':
      return windingNumber !== 0;
    case 'evenodd':
      return windingNumber % 2 !== 0;
    case 'positive':
      return windingNumber > 0;
    case 'negative':
      return windingNumber < 0;
    default:
      throw new Error( 'unknown fill rule' );
  }
};
