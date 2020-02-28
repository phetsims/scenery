// Copyright 2013-2020, University of Colorado Boulder


/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import inherit from '../../../phet-core/js/inherit.js';
import scenery from '../scenery.js';
import Drawable from './Drawable.js';

function InlineCanvasCacheDrawable( renderer, instance ) {
  Drawable.call( this, renderer );


  //OHTWO TODO: pooling!

  // TODO: NOTE: may have to separate into separate drawables for separate group renderers

  this.instance = instance; // will need this so we can get bounds for layer fitting
}

scenery.register( 'InlineCanvasCacheDrawable', InlineCanvasCacheDrawable );

inherit( Drawable, InlineCanvasCacheDrawable, {
  // TODO: support Canvas/SVG/DOM

  stitch: function( firstDrawable, lastDrawable, firstChangeInterval, lastChangeInterval ) {
    //OHTWO TODO: called when we have a change in our drawables
  }
} );

export default InlineCanvasCacheDrawable;