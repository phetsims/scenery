// Copyright 2013-2016, University of Colorado Boulder

/**
 * TODO docs
 *   note paintCanvas() required, and other implementation-specific details
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var ExperimentalPoolable = require( 'PHET_CORE/ExperimentalPoolable' );
  var inherit = require( 'PHET_CORE/inherit' );
  var PaintableStatelessDrawable = require( 'SCENERY/display/drawables/PaintableStatelessDrawable' );
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

    update: function() {
      this.dirty = false;
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

  // methods for forwarding dirty messages
  function canvasSelfDirty() {
    // we pass this method and it is only called with blah.call( ... ), where the 'this' reference is set.
    this.markDirty();
  }

  // options takes: type, paintCanvas( wrapper ), usesPaint, and dirtyMethods (array of string names of methods that make the state dirty)
  CanvasSelfDrawable.createDrawable = function( options ) {
    var type = options.type;
    var paintCanvas = options.paintCanvas;
    var usesPaint = options.usesPaint;

    assert && assert( typeof type === 'function' );
    assert && assert( typeof paintCanvas === 'function' );
    assert && assert( typeof usesPaint === 'boolean' );

    inherit( CanvasSelfDrawable, type, {
      initialize: function( renderer, instance ) {
        this.initializeCanvasSelfDrawable( renderer, instance );

        if ( usesPaint ) {
          this.initializePaintableStateless( renderer, instance );
        }

        return this; // allow for chaining
      },

      paintCanvas: paintCanvas,

      update: function() {
        // no action directly needed for the self-drawable case, as we will be repainted in the block
        this.dirty = false;
      },

      dispose: function() {
        CanvasSelfDrawable.prototype.dispose.call( this );

        if ( usesPaint ) {
          this.disposePaintableStateless();
        }
      }
    } );

    // include stubs (stateless) for marking dirty stroke and fill (if necessary). we only want one dirty flag, not multiple ones, for Canvas (for now)
    if ( usesPaint ) {
      PaintableStatelessDrawable.mixInto( type );
    }

    // set up pooling
    ExperimentalPoolable.mixInto( type, {
      initialize: type.prototype.initialize
    } );

    if ( options.dirtyMethods ) {
      for ( var i = 0; i < options.dirtyMethods.length; i++ ) {
        type.prototype[ options.dirtyMethods[ i ] ] = canvasSelfDirty;
      }
    }

    return type;
  };

  return CanvasSelfDrawable;
} );
