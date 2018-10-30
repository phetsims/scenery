// Copyright 2013-2016, University of Colorado Boulder

/**
 * Represents an SVG visual element, and is responsible for tracking changes to the visual element, and then applying
 * any changes at a later time.
 *
 * Abstract methods to implement for concrete implementations:
 *   updateSVGSelf() - Update the SVG element's state to what the Node's self should display
 *   updateDefsSelf( block ) - Update defs on the given block (or if block === null, remove)
 *   initializeState( renderer, instance )
 *   disposeState()
 *
 * Subtypes should also implement drawable.svgElement, as the actual SVG element to be used.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var PaintSVGState = require( 'SCENERY/display/PaintSVGState' );
  var scenery = require( 'SCENERY/scenery' );
  var SelfDrawable = require( 'SCENERY/display/SelfDrawable' );

  function SVGSelfDrawable( renderer, instance ) {
    this.initializeSVGSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  }

  scenery.register( 'SVGSelfDrawable', SVGSelfDrawable );

  inherit( SelfDrawable, SVGSelfDrawable, {
    initializeSVGSelfDrawable: function( renderer, instance, usesPaint, keepElements ) {
      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      this.usesPaint = usesPaint;
      this.keepElements = keepElements;

      this.svgElement = null; // should be filled in by subtype
      this.svgBlock = null; // will be updated by updateSVGBlock()

      this.initializeState( renderer, instance ); // assumes we have a state trait

      if ( this.usesPaint ) {
        if ( !this.paintState ) {
          this.paintState = new PaintSVGState();
        }
        else {
          this.paintState.initialize();
        }
      }

      return this; // allow chaining
    },

    /**
     * Updates the DOM appearance of this drawable (whether by preparing/calling draw calls, DOM element updates, etc.)
     * @public
     * @override
     *
     * @returns {boolean} - Whether the update should continue (if false, further updates in supertype steps should not
     *                      be done).
     */
    update: function() {
      // See if we need to actually update things (will bail out if we are not dirty, or if we've been disposed)
      if ( !SelfDrawable.prototype.update.call( this ) ) {
        return false;
      }

      this.updateSVG();

      return true;
    },

    // @protected: called to update the visual appearance of our svgElement
    updateSVG: function() {
      if ( this.paintDirty ) {
        this.updateSVGSelf( this.node, this.svgElement );
      }

      // sync the differences between the previously-recorded list of cached paints and the new list
      if ( this.usesPaint && this.dirtyCachedPaints ) {
        var newCachedPaints = this.node._cachedPaints.slice(); // defensive copy for now
        var i;
        var j;

        // scan for new cached paints (not in the old list)
        for ( i = 0; i < newCachedPaints.length; i++ ) {
          var newPaint = newCachedPaints[ i ];
          var isNew = true;
          for ( j = 0; j < this.lastCachedPaints.length; j++ ) {
            if ( newPaint === this.lastCachedPaints[ j ] ) {
              isNew = false;
              break;
            }
          }
          if ( isNew ) {
            this.svgBlock.incrementPaint( newPaint );
          }
        }
        // scan for removed cached paints (not in the new list)
        for ( i = 0; i < this.lastCachedPaints.length; i++ ) {
          var oldPaint = this.lastCachedPaints[ i ];
          var isRemoved = true;
          for ( j = 0; j < newCachedPaints.length; j++ ) {
            if ( oldPaint === newCachedPaints[ j ] ) {
              isRemoved = false;
              break;
            }
          }
          if ( isRemoved ) {
            this.svgBlock.decrementPaint( oldPaint );
          }
        }

        this.lastCachedPaints = newCachedPaints;
      }

      // clear all of the dirty flags
      this.setToCleanState();
    },

    // to be used by our passed in options.updateSVG
    updateFillStrokeStyle: function( element ) {
      if ( !this.usesPaint ) {
        return;
      }

      if ( this.dirtyFill ) {
        this.paintState.updateFill( this.svgBlock, this.node.getFillValue() );
      }
      if ( this.dirtyStroke ) {
        this.paintState.updateStroke( this.svgBlock, this.node.getStrokeValue() );
      }
      var strokeDetailDirty = this.dirtyLineWidth || this.dirtyLineOptions;
      if ( strokeDetailDirty ) {
        this.paintState.updateStrokeDetailStyle( this.node );
      }
      if ( this.dirtyFill || this.dirtyStroke || strokeDetailDirty ) {
        element.setAttribute( 'style', this.paintState.baseStyle + this.paintState.strokeDetailStyle );
      }

      this.cleanPaintableState();
    },

    updateSVGBlock: function( svgBlock ) {
      // remove cached paint references from the old svgBlock
      var oldSvgBlock = this.svgBlock;
      if ( this.usesPaint && oldSvgBlock ) {
        for ( var i = 0; i < this.lastCachedPaints.length; i++ ) {
          oldSvgBlock.decrementPaint( this.lastCachedPaints[ i ] );
        }
      }

      this.svgBlock = svgBlock;

      // add cached paint references from the new svgBlock
      if ( this.usesPaint ) {
        for ( var j = 0; j < this.lastCachedPaints.length; j++ ) {
          svgBlock.incrementPaint( this.lastCachedPaints[ j ] );
        }
      }

      this.updateDefsSelf && this.updateDefsSelf( svgBlock );

      this.usesPaint && this.paintState.updateSVGBlock( svgBlock );

      // since fill/stroke IDs may be block-specific, we need to mark them dirty so they will be updated
      this.usesPaint && this.markDirtyFill();
      this.usesPaint && this.markDirtyStroke();
    },

    dispose: function() {
      this.disposeState(); // assumes subtype existence

      if ( !this.keepElements ) {
        // clear the references
        this.svgElement = null;
      }

      // release any defs, and dispose composed state objects
      this.updateDefsSelf && this.updateDefsSelf( null );
      this.usesPaint && this.paintState.dispose();

      this.defs = null;

      this.svgBlock = null;

      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  return SVGSelfDrawable;
} );
