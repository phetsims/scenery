// Copyright 2024, University of Colorado Boulder

/**
 * Query parameters for the scenery library.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

import scenery from './scenery.js';

const sceneryQueryParameters = QueryStringMachine.getAll( {

  /**
   * If this is a finite number AND assertions are enabled, it will track maximum Node parent counts, and
   * will assert that the count is not greater than the limit.
   */
  parentLimit: {
    type: 'number',
    defaultValue: Number.POSITIVE_INFINITY,
    public: false
  },

  /**
   * If this is a finite number AND assertions are enabled, it will track maximum Node child counts, and
   * will assert that the number of children on a single Node is not greater than the limit.
   */
  childLimit: {
    type: 'number',
    defaultValue: Number.POSITIVE_INFINITY,
    public: false
  }
} );

scenery.register( 'sceneryQueryParameters', sceneryQueryParameters );

export default sceneryQueryParameters;