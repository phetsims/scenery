// Copyright 2013-2025, University of Colorado Boulder

/**
 * TODO docs https://github.com/phetsims/scenery/issues/1581
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Drawable from '../display/Drawable.js';
import scenery from '../scenery.js';

class SharedCanvasCacheDrawable extends Drawable {
  /**
   * @param {Trail} trail
   * @param {number} renderer
   * @param {Instance} instance
   * @param {Instance} sharedInstance
   */
  constructor( trail, renderer, instance, sharedInstance ) {
    super();

    this.initialize( trail, renderer, instance, sharedInstance );
  }

  /**
   * @public
   * @override
   *
   * @param {Trail} trail
   * @param {number} renderer
   * @param {Instance} instance
   * @param {Instance} sharedInstance
   */
  initialize( trail, renderer, instance, sharedInstance ) {
    super.initialize( renderer );

    // TODO: NOTE: may have to separate into separate drawables for separate group renderers https://github.com/phetsims/scenery/issues/1581

    this.instance = instance; // will need this so we can get bounds for layer fitting
    this.sharedInstance = sharedInstance;
  }
}

scenery.register( 'SharedCanvasCacheDrawable', SharedCanvasCacheDrawable );
export default SharedCanvasCacheDrawable;