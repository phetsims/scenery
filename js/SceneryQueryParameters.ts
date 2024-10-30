// Copyright 2024, University of Colorado Boulder

/**
 * Query parameters for the scenery library.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 */

/* global QSMParsedParameters QueryStringMachineSchema */

import scenery from './scenery.js';

// All scenery query parameters MUST have a default value, since we do not always support QSM as a global.
const schema = {

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
} satisfies Record<string, QueryStringMachineSchema>;

// Scenery doesn't depend on QSM, so be graceful here, and take default values.
const sceneryQueryParameters = window.hasOwnProperty( 'QueryStringMachine' ) ?
                               QueryStringMachine.getAll( schema ) :
                               ( Object.keys( schema ) as ( keyof typeof schema )[] ).map( key => {
                                 return { [ key ]: schema[ key ].defaultValue };
                               } );

scenery.register( 'sceneryQueryParameters', sceneryQueryParameters );

export default sceneryQueryParameters as QSMParsedParameters<typeof schema>;