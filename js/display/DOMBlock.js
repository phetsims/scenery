// Copyright 2002-2014, University of Colorado Boulder


/**
 * DOM Drawable wrapper for another DOM Drawable. Used so that we can have our own independent siblings, generally as part
 * of a Backbone's layers/blocks.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var PoolableMixin = require( 'PHET_CORE/PoolableMixin' );
  var scenery = require( 'SCENERY/scenery' );
  var Block = require( 'SCENERY/display/Block' );

  scenery.DOMBlock = function DOMBlock( display, domDrawable ) {
    this.initialize( display, domDrawable );
  };
  var DOMBlock = scenery.DOMBlock;

  inherit( Block, DOMBlock, {
    initialize: function( display, domDrawable ) {
      // TODO: is it bad to pass the acceleration flags along?
      this.initializeBlock( display, domDrawable.renderer );

      this.domDrawable = domDrawable;
      this.domElement = domDrawable.domElement;

      return this;
    },

    dispose: function() {
      this.domDrawable = null;
      this.domElement = null;

      // super call
      Block.prototype.dispose.call( this );
    },

    update: function() {
      if ( this.dirty && !this.disposed ) {
        this.dirty = false;

        this.domDrawable.update();
      }
    },

    markDirtyDrawable: function( drawable ) {
      this.markDirty();
    },

    addDrawable: function( drawable ) {
      sceneryLog && sceneryLog.DOMBlock && sceneryLog.DOMBlock( '#' + this.id + '.addDrawable ' + drawable.toString() );
      assert && assert( this.domDrawable === drawable, 'DOMBlock should only be used with one drawable for now (the one it was initialized with)' );

      Block.prototype.addDrawable.call( this, drawable );
    },

    removeDrawable: function( drawable ) {
      sceneryLog && sceneryLog.DOMBlock && sceneryLog.DOMBlock( '#' + this.id + '.removeDrawable ' + drawable.toString() );
      assert && assert( this.domDrawable === drawable, 'DOMBlock should only be used with one drawable for now (the one it was initialized with)' );

      Block.prototype.removeDrawable.call( this, drawable );
    }
  } );

  /* jshint -W064 */
  PoolableMixin( DOMBlock, {
    constructorDuplicateFactory: function( pool ) {
      return function( display, domDrawable ) {
        if ( pool.length ) {
          return pool.pop().initialize( display, domDrawable );
        }
        else {
          return new DOMBlock( display, domDrawable );
        }
      };
    }
  } );

  return DOMBlock;
} );
