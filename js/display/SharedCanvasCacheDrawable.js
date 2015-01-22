// Copyright 2002-2014, University of Colorado Boulder


/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );

  scenery.SharedCanvasCacheDrawable = function SharedCanvasCacheDrawable( trail, renderer, instance, sharedInstance ) {
    Drawable.call( this, renderer );

    // TODO: NOTE: may have to separate into separate drawables for separate group renderers

    this.instance = instance; // will need this so we can get bounds for layer fitting
    this.sharedInstance = sharedInstance;
  };
  var SharedCanvasCacheDrawable = scenery.SharedCanvasCacheDrawable;

  inherit( Drawable, SharedCanvasCacheDrawable, {
    // TODO: support Canvas/SVG/DOM
  } );

  return SharedCanvasCacheDrawable;
} );
