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

  scenery.WebGLSelfDrawable = function WebGLSelfDrawable( renderer, instance ) {
    this.initializeWebGLSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  };
  var WebGLSelfDrawable = scenery.WebGLSelfDrawable;

  inherit( SelfDrawable, WebGLSelfDrawable, {
    initializeWebGLSelfDrawable: function( renderer, instance ) {
      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      // this is the same across lifecycles
      this.transformListener = this.transformListener || this.markTransformDirty.bind( this );

      instance.addRelativeTransformListener( this.transformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
      instance.addRelativeTransformPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated

      return this;
    },

    markTransformDirty: function() {
      this.markDirty();
    },

    dispose: function() {
      this.instance.removeRelativeTransformListener( this.transformListener );
      this.instance.removeRelativeTransformPrecompute();

      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  return WebGLSelfDrawable;
} );
