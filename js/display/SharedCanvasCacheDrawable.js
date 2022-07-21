// Copyright 2013-2022, University of Colorado Boulder

/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Drawable, scenery } from '../imports.js';

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

    // TODO: NOTE: may have to separate into separate drawables for separate group renderers

    this.instance = instance; // will need this so we can get bounds for layer fitting
    this.sharedInstance = sharedInstance;
  }
}

scenery.register( 'SharedCanvasCacheDrawable', SharedCanvasCacheDrawable );
export default SharedCanvasCacheDrawable;