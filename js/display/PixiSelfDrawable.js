// Copyright 2002-2014, University of Colorado Boulder


/**
 * Represents a Pixi visual element, and is responsible for tracking changes to the visual element, and then applying
 * any changes at a later time.
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
    initializePixiSelfDrawable: function( renderer, instance, keepElements ) {
      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      this.keepElements = keepElements;

      this.displayObject = null; // should be filled in by subtype
      this.paintDirty = true;

      return this;
    },

    // @public: called from elsewhere to update the SVG element
    update: function() {
      if ( this.dirty ) {
        this.dirty = false;

        this.updatePixiSelf.call( this, this.node, this.displayObject );
      }
    },

    // general flag set on the state, which we forward directly to the drawable's paint flag
    markPaintDirty: function() {
      this.markDirty();
    },

    dispose: function() {
      if ( !this.keepElements ) {
        // clear the references
        this.displayObject = null;
      }

      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  return PixiSelfDrawable;
} );
