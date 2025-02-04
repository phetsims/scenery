// Copyright 2013-2025, University of Colorado Boulder

/**
 * TODO docs https://github.com/phetsims/scenery/issues/1581
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Drawable from '../display/Drawable.js';
import scenery from '../scenery.js';

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

    // TODO: NOTE: may have to separate into separate drawables for separate group renderers https://github.com/phetsims/scenery/issues/1581

    // @public {Instance}
    this.instance = instance; // will need this so we can get bounds for layer fitting
  }

  // TODO: support Canvas/SVG/DOM https://github.com/phetsims/scenery/issues/1581

  /**
   * @public
   *
   * @param {Drawable} firstDrawable
   * @param {Drawable} lastDrawable
   * @param {ChangeInterval} firstChangeInterval
   * @param {ChangeInterval} lastChangeInterval
   */
  stitch( firstDrawable, lastDrawable, firstChangeInterval, lastChangeInterval ) {
    //OHTWO TODO: called when we have a change in our drawables https://github.com/phetsims/scenery/issues/1581
  }
}

scenery.register( 'InlineCanvasCacheDrawable', InlineCanvasCacheDrawable );
export default InlineCanvasCacheDrawable;