// Copyright 2013-2019, University of Colorado Boulder


/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( require => {
  'use strict';

  const Drawable = require( 'SCENERY/display/Drawable' );
  const inherit = require( 'PHET_CORE/inherit' );
  const scenery = require( 'SCENERY/scenery' );

  function SharedCanvasCacheDrawable( trail, renderer, instance, sharedInstance ) {
    Drawable.call( this, renderer );

    // TODO: NOTE: may have to separate into separate drawables for separate group renderers

    this.instance = instance; // will need this so we can get bounds for layer fitting
    this.sharedInstance = sharedInstance;
  }

  scenery.register( 'SharedCanvasCacheDrawable', SharedCanvasCacheDrawable );

  inherit( Drawable, SharedCanvasCacheDrawable, {
    // TODO: support Canvas/SVG/DOM
  } );

  return SharedCanvasCacheDrawable;
} );
