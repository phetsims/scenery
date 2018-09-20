// Copyright 2013-2015, University of Colorado Boulder


/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var Drawable = require( 'SCENERY/display/Drawable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

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

  return InlineCanvasCacheDrawable;
} );
