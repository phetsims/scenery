// Copyright 2002-2014, University of Colorado Boulder


/**
 * Represents an SVG visual element, and is responsible for tracking changes to the visual element, and then applying any changes at a later time.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
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

      this.displayObject = null; // should be filled in by subtype

      return this;
    },

    // @public: called from elsewhere to update the SVG element
    update: function() {
      if ( this.dirty ) {
        this.dirty = false;
        this.updatePixi();
      }
    },

    // @protected: called to update the visual appearance of our svgElement
    updatePixi: function() {
      // should generally be overridden by drawable subtypes to implement the update
    },

    dispose: function() {
      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  /*
   * Options contains:
   *   type - the constructor, should be of the form: function SomethingSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); }.
   *          Used for debugging constructor name.
   *   stateType - function to apply to mix-in the state (TODO docs)
   *   initialize( renderer, instance ) - should initialize this.svgElement if it doesn't already exist, and set up any other initial state properties
   *   updateSVG() - updates the svgElement to the latest state recorded
   *   updateSVGBlock( svgBlock ) - called when the SVGBlock object needs to be switched (or initialized)
   *   usesPaint - whether we include paintable (fill/stroke) state & defs
   *   keepElements - when disposing a drawable (not used anymore), should we keep a reference to the SVG element so we don't have to recreate it when reinitialized?
   */
  PixiSelfDrawable.createDrawable = function( options ) {
    var type = options.type;
    var stateType = options.stateType;
    var initializeSelf = options.initialize;
    var updatePixiSelf = options.updatePixi;
    var usesPaint = options.usesPaint;
    var keepElements = options.keepElements;

    assert && assert( typeof type === 'function' );
    assert && assert( typeof stateType === 'function' );
    assert && assert( typeof initializeSelf === 'function' );
    assert && assert( typeof updatePixiSelf === 'function' );
    assert && assert( typeof usesPaint === 'boolean' );
    assert && assert( typeof keepElements === 'boolean' );

    inherit( PixiSelfDrawable, type, {
      initialize: function( renderer, instance ) {
        this.initializePixiSelfDrawable( renderer, instance );

        initializeSelf.call( this, renderer, instance );

        return this; // allow for chaining
      },

      updatePixi: function() {
        if ( this.paintDirty ) {
          updatePixiSelf.call( this, this.node, this.displayObject );
        }

        // clear all of the dirty flags
        this.setToClean();
      },

      onAttach: function( node ) {

      },

      // release the SVG elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
      onDetach: function( node ) {
        if ( !keepElements ) {
          // clear the references
          this.displayObject = null;
        }
      },

      setToClean: function() {
        this.setToCleanState();
      }
    } );

    // mix-in
    stateType( type );

    // set up pooling
    SelfDrawable.Poolable.mixin( type );

    return type;
  };

  return PixiSelfDrawable;
} );
