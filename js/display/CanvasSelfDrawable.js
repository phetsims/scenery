// Copyright 2013-2016, University of Colorado Boulder

/**
 * TODO docs
 *   note paintCanvas() required, and other implementation-specific details
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );

  /**
   * @constructor
   * @mixes Poolable
   *
   * @param renderer
   * @param instance
   */
  function CanvasSelfDrawable( renderer, instance ) {
    this.initializeCanvasSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  }

  scenery.register( 'CanvasSelfDrawable', CanvasSelfDrawable );

  inherit( SelfDrawable, CanvasSelfDrawable, {
    initializeCanvasSelfDrawable: function( renderer, instance ) {
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

    // general flag set on the state, which we forward directly to the drawable's paint flag
    markPaintDirty: function() {
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

  return CanvasSelfDrawable;
} );
