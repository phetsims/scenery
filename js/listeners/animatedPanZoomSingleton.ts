// Copyright 2021-2022, University of Colorado Boulder

/**
 * A singleton with an instance of an AnimatedPanZoomListener that can be easily accessed for an application.
 * The AnimatedPanZoomListener adds global key press listeners and it is reasonable that you would only want
 * one for a document/application.
 *
 * @author Jesse Greenberg
 */

import { AnimatedPanZoomListener, scenery } from '../imports.js';

class AnimatedPanZoomSingleton {
  constructor() {

    // @private {null|AnimatedPanZoomListener} - null until initialized
    this._listener = null;
  }

  /**
   * @public
   * @param {../nodes/Node} targetNode
   * @param {Object} [options]
   */
  initialize( targetNode, options ) {
    this._listener = new AnimatedPanZoomListener( targetNode, options );
  }

  /**
   * @public
   */
  dispose() {
    this._listener.dispose();
    this._listener = null;
  }

  /**
   * Returns the AnimatedPanZoomListener.
   * @public
   *
   * @returns {null|AnimatedPanZoomListener}
   */
  get listener() {
    assert && assert( this._listener, 'No listener, call initialize first.' );
    return this._listener;
  }

  /**
   * Returns whether or not the animatedPanZoomSingleton has been initialized.
   * @returns {boolean}
   */
  get initialized() {
    return !!this._listener;
  }
}

const animatedPanZoomSingleton = new AnimatedPanZoomSingleton();
scenery.register( 'animatedPanZoomSingleton', animatedPanZoomSingleton );
export default animatedPanZoomSingleton;