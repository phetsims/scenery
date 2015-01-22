// Copyright 2002-2014, University of Colorado Boulder

/**
 * TODO docs
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 * @author Sam Reid
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );

  scenery.PixiSelfDrawable = function PixiSelfDrawable( renderer, instance ) {
    this.initializePixiSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  };
  var PixiSelfDrawable = scenery.PixiSelfDrawable;

  inherit( SelfDrawable, PixiSelfDrawable, {
    initializePixiSelfDrawable: function( renderer, instance ) {
      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      // this is the same across lifecycles
      this.transformListener = this.transformListener || this.markTransformDirty.bind( this );

      instance.relativeTransform.addListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
      instance.relativeTransform.addPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated

      return this;
    },

    markTransformDirty: function() {
      this.markDirty();
    },

    dispose: function() {
      this.instance.relativeTransform.removeListener( this.transformListener );
      this.instance.relativeTransform.removePrecompute();

      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  return PixiSelfDrawable;
} );
