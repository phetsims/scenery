// Copyright 2020, University of Colorado Boulder

/**
 * Base type for filters
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import scenery from '../scenery.js';

let globalId = 1;

class Filter {
  constructor() {
    // @public (scenery-internal) {string}
    this.id = 'filter' + globalId++;
  }

  /**
   * @public
   * @abstract
   *
   * @returns {string}
   */
  getCSSFilterString() {
    throw new Error( 'abstract method' );
  }
}

scenery.register( 'Filter', Filter );
export default Filter;