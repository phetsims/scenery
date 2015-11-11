// Copyright 2013-2015, University of Colorado Boulder

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

  function WebGLSelfDrawable( renderer, instance ) {
    this.initializeWebGLSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  }

  scenery.register( 'WebGLSelfDrawable', WebGLSelfDrawable );

  inherit( SelfDrawable, WebGLSelfDrawable, {
    initializeWebGLSelfDrawable: function( renderer, instance ) {
      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      // this is the same across lifecycles
      this.transformListener = this.transformListener || this.markTransformDirty.bind( this );

      // when our relative transform changes, notify us in the pre-repaint phase
      instance.relativeTransform.addListener( this.transformListener );

      // trigger precomputation of the relative transform, since we will always need it when it is updated
      instance.relativeTransform.addPrecompute();

      return this;
    },

    markTransformDirty: function() {
      this.markDirty();
    },

    // @override
    updateSelfVisibility: function() {
      SelfDrawable.prototype.updateSelfVisibility.call( this );

      // mark us as dirty when our self visibility changes
      this.markDirty();
    },

    dispose: function() {
      this.instance.relativeTransform.removeListener( this.transformListener );
      this.instance.relativeTransform.removePrecompute();

      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  return WebGLSelfDrawable;
} );
