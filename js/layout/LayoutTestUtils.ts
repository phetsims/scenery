// Copyright 2024-2025, University of Colorado Boulder


/**
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import Utils from '../../../dot/js/Utils.js';
import Rectangle from '../nodes/Rectangle.js';
import type { RectangleOptions } from '../nodes/Rectangle.js';

const RECT_WIDTH = 100;
const RECT_HEIGHT = 25;

const LayoutTestUtils = {

  RECT_WIDTH: RECT_WIDTH,
  RECT_HEIGHT: RECT_HEIGHT,

  createRectangles: ( count: number, indexToOptions?: ( index: number ) => RectangleOptions ): Rectangle[] => {
    return _.times( count, ( index: number ) => {
      const options = indexToOptions ? indexToOptions( index ) : {};
      return new Rectangle( 0, 0, RECT_WIDTH, RECT_HEIGHT, options );
    } );
  },

  /**
   * Convenience method for comparing two numbers with an epsilon. Layout calculations often have floating point
   * errors, so this method is useful for comparing two numbers that should be equal within a small epsilon.
   */
  aboutEqual( a: number, b: number, epsilon = 0.0001 ): boolean {
    return Utils.equalsEpsilon( a, b, epsilon );
  }
};

export default LayoutTestUtils;