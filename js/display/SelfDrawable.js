// Copyright 2014-2022, University of Colorado Boulder


/**
 * A drawable that will paint a single instance.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Drawable, scenery } from '../imports.js';

class SelfDrawable extends Drawable {
  /**
   * We have enough concrete types that want a fallback constructor to this, so we'll provide it for convenience.
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  constructor( renderer, instance ) {
    assert && assert( typeof renderer === 'number' );
    assert && assert( instance );

    super();

    this.initialize( renderer, instance );
  }

  /**
   * @public
   *
   * @param {number} renderer
   * @param {Instance} instance
   * @returns {SelfDrawable}
   */
  initialize( renderer, instance ) {
    super.initialize( renderer );

    // @private {function}
    this.drawableVisibilityListener = this.drawableVisibilityListener || this.updateSelfVisibility.bind( this );

    // @public {Instance}
    this.instance = instance;

    // @public {Node}
    this.node = instance.trail.lastNode();
    this.node.attachDrawable( this );

    this.instance.selfVisibleEmitter.addListener( this.drawableVisibilityListener );

    this.updateSelfVisibility();

    return this;
  }

  /**
   * Releases references
   * @public
   * @override
   */
  dispose() {
    this.instance.selfVisibleEmitter.removeListener( this.drawableVisibilityListener );

    this.node.detachDrawable( this );

    // free references
    this.instance = null;
    this.node = null;

    super.dispose();
  }

  /**
   * @public
   */
  updateSelfVisibility() {
    // hide our drawable if it is not relatively visible
    this.visible = this.instance.selfVisible;
  }

  /**
   * Returns a more-informative string form of this object.
   * @public
   * @override
   *
   * @returns {string}
   */
  toDetailedString() {
    return `${this.toString()} (${this.instance.trail.toPathString()})`;
  }
}

scenery.register( 'SelfDrawable', SelfDrawable );

export default SelfDrawable;