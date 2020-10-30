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

  /**
   * @public
   *
   * @returns {boolean}
   */
  isDOMCompatible() {
    // TODO: We can browser-check on things like color matrix? But we want to disallow things that we can't guarantee we
    // can support?
    return false;
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  isSVGCompatible() {
    return false;
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  isCanvasCompatible() {
    return false;
  }

  /**
   * @public
   *
   * @returns {boolean}
   */
  isWebGLCompatible() {
    return false;
  }
}

scenery.register( 'Filter', Filter );
export default Filter;