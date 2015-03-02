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
  var Paintable = require( 'SCENERY/nodes/Paintable' );

  scenery.SVGSelfDrawable = function SVGSelfDrawable( renderer, instance ) {
    this.initializeSVGSelfDrawable( renderer, instance );

    throw new Error( 'Should use initialization and pooling' );
  };
  var SVGSelfDrawable = scenery.SVGSelfDrawable;

  inherit( SelfDrawable, SVGSelfDrawable, {
    initializeSVGSelfDrawable: function( renderer, instance ) {
      // super initialization
      this.initializeSelfDrawable( renderer, instance );

      this.svgElement = null; // should be filled in by subtype
      this.svgBlock = null; // will be updated by updateSVGBlock()

      return this;
    },

    // @public: called when the defs block changes
    // NOTE: should generally be overridden by drawable subtypes, so they can apply their defs changes
    updateSVGBlock: function( svgBlock ) {
      this.svgBlock = svgBlock;
    },

    // @public: called from elsewhere to update the SVG element
    update: function() {
      if ( this.dirty ) {
        this.dirty = false;
        this.updateSVG();
      }
    },

    // @protected: called to update the visual appearance of our svgElement
    updateSVG: function() {
      // should generally be overridden by drawable subtypes to implement the update
    },

    dispose: function() {
      this.svgBlock = null;

      SelfDrawable.prototype.dispose.call( this );
    }
  } );

  /*
   * Options contains:
   *   type - the constructor, should be of the form: function SomethingSVGDrawable( renderer, instance ) { this.initialize( renderer, instance ); }.
   *          Used for debugging constructor name.
   *   initialize( renderer, instance ) - should initialize this.svgElement if it doesn't already exist, and set up any other initial state properties
   *   updateSVG() - updates the svgElement to the latest state recorded
   *   updateSVGBlock( svgBlock ) - called when the SVGBlock object needs to be switched (or initialized)
   *   usesPaint - whether we include paintable (fill/stroke) state & defs
   *   keepElements - when disposing a drawable (not used anymore), should we keep a reference to the SVG element so we don't have to recreate it when reinitialized?
   */
  SVGSelfDrawable.createDrawable = function( options ) {
    var type = options.type;
    var initializeSelf = options.initialize;
    var updateSVGSelf = options.updateSVG;
    var updateDefsSelf = options.updateDefs;
    var usesPaint = options.usesPaint;
    var keepElements = options.keepElements;

    assert && assert( typeof type === 'function' );
    assert && assert( typeof initializeSelf === 'function' );
    assert && assert( typeof updateSVGSelf === 'function' );
    assert && assert( !updateDefsSelf || ( typeof updateDefsSelf === 'function' ) );
    assert && assert( typeof usesPaint === 'boolean' );
    assert && assert( typeof keepElements === 'boolean' );

    inherit( SVGSelfDrawable, type, {
      initialize: function( renderer, instance ) {
        this.initializeSVGSelfDrawable( renderer, instance );
        this.initializeState(); // assumes we have a state mixin

        initializeSelf.call( this, renderer, instance );

        // tracks our current svgBlock object, so we can update our fill/stroke/etc. on our own
        this.svgBlock = null;

        if ( usesPaint ) {
          if ( !this.paintState ) {
            this.paintState = new Paintable.PaintSVGState();
          }
          else {
            this.paintState.initialize();
          }
        }

        return this; // allow for chaining
      },

      // to be used by our passed in options.updateSVG
      updateFillStrokeStyle: function( element ) {
        if ( !usesPaint ) {
          return;
        }

        if ( this.dirtyFill ) {
          this.paintState.updateFill( this.svgBlock, this.node._fill );
        }
        if ( this.dirtyStroke ) {
          this.paintState.updateStroke( this.svgBlock, this.node._stroke );
        }
        var strokeParameterDirty = this.dirtyLineWidth || this.dirtyLineOptions;
        if ( strokeParameterDirty ) {
          this.paintState.updateStrokeParameters( this.node );
        }
        if ( this.dirtyFill || this.dirtyStroke || strokeParameterDirty ) {
          element.setAttribute( 'style', this.paintState.baseStyle + this.paintState.extraStyle );
        }
      },

      updateSVG: function() {
        if ( this.paintDirty ) {
          updateSVGSelf.call( this, this.node, this.svgElement );
        }

        // sync the differences between the previously-recorded list of cached paints and the new list
        if ( usesPaint && this.dirtyCachedPaints ) {
          var newCachedPaints = this.node._cachedPaints.slice(); // defensive copy for now
          var i, j;
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
        this.setToClean();
      },

      updateSVGBlock: function( svgBlock ) {
        // remove cached paint references from the old svgBlock
        var oldSvgBlock = this.svgBlock;
        if ( usesPaint && oldSvgBlock ) {
          for ( var i = 0; i < this.lastCachedPaints.length; i++ ) {
            oldSvgBlock.decrementPaint( this.lastCachedPaints[ i ] );
          }
        }

        this.svgBlock = svgBlock;

        // add cached paint references from the new svgBlock
        if ( usesPaint ) {
          for ( var j = 0; j < this.lastCachedPaints.length; j++ ) {
            svgBlock.incrementPaint( this.lastCachedPaints[ j ] );
          }
        }

        updateDefsSelf && updateDefsSelf.call( this, svgBlock );

        usesPaint && this.paintState.updateSVGBlock( svgBlock );

        // since fill/stroke IDs may be block-specific, we need to mark them dirty so they will be updated
        usesPaint && this.markDirtyFill();
        usesPaint && this.markDirtyStroke();
      },

      onAttach: function( node ) {

      },

      // release the SVG elements from the poolable visual state so they aren't kept in memory. May not be done on platforms where we have enough memory to pool these
      onDetach: function( node ) {
        //OHTWO TODO: are we missing the disposal?
        if ( !keepElements ) {
          // clear the references
          this.svgElement = null;
        }

        // release any defs, and dispose composed state objects
        updateDefsSelf && updateDefsSelf.call( this, null );
        usesPaint && this.paintState.dispose();

        this.defs = null;
      },

      setToClean: function() {
        this.setToCleanState();
      }
    } );

    // set up pooling
    SelfDrawable.Poolable.mixin( type );

    return type;
  };

  return SVGSelfDrawable;
} );
