// Copyright 2013-2022, University of Colorado Boulder

/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Drawable, scenery } from '../imports.js';

class InlineCanvasCacheDrawable extends Drawable {
  /**
   * @param {number} renderer
   * @param {Instance} instance
   */
  constructor( renderer, instance ) {
    super();

    this.initialize( renderer, instance );
  }

  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer );

    // TODO: NOTE: may have to separate into separate drawables for separate group renderers

    // @public {Instance}
    this.instance = instance; // will need this so we can get bounds for layer fitting
  }

  // TODO: support Canvas/SVG/DOM

  /**
   * @public
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   * @param {ChangeInterval} firstChangeInterval
   * @param {ChangeInterval} lastChangeInterval
   */
  stitch( firstDrawable, lastDrawable, firstChangeInterval, lastChangeInterval ) {
    //OHTWO TODO: called when we have a change in our drawables
  }
}

scenery.register( 'InlineCanvasCacheDrawable', InlineCanvasCacheDrawable );
export default InlineCanvasCacheDrawable;