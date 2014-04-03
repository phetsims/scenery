// Copyright 2002-2013, University of Colorado

/**
 * Represents an SVG visual element, and is responsible for tracking changes to the visual element, and then applying any changes at a later time.
 *
 * Node API needed:
 * {
 *   attachSVGDrawable: function( SVGSelfDrawable )
 *   detachSVGDrawable: function( SVGSelfDrawable )
 * }
 *
 * visual state API needed:
 * {
 *   markPaintDirty: function()   // used by Strokable/Fillable
 *   node: Node                   // used by Strokable/Fillable
 *   drawable: SVGSelfDrawable    // set by the visual state on initialization     NOTE: required for any type of visual state! (used by Strokable/Fillable states)
 *   svgElement: SVGElement       // what is displayed in the SVG tree (should be the base of the displayed element)
 *   updateSVG: function()        // updates any visual state
 *   updateDefs: function( defs ) // a change to where the defs are stored occurred. Passed in is the new defs block
 *   onDetach: function()         // called when the state is detached from a drawable. optionally discard DOM elements. we guarantee state will be
 *                                // initialized again before any more update() calls
 * }
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );
  var Drawable = require( 'SCENERY/display/Drawable' );
  
  scenery.SVGSelfDrawable = function SVGSelfDrawable( trail, renderer, instance ) {
    Drawable.call( this, renderer );
    this.instance = instance;
    
    this.node = instance.trail.lastNode();
    
    this.visualState = null; // to be created in attachSVGDrawable
    this.dirty = true;
    
    this.node.attachSVGDrawable( this ); // should set this.visualState
    
    // now that we called attachSVGDrawable, update the visualState object with the flags it will need
    this.svgElement = this.visualState.svgElement;
    
    this.defs = null; // TODO NOTE: stub spot fot defs. if we support changing defs locations, we'll need to redo a bit of API (this is accessed by visual states)
  };
  var SVGSelfDrawable = scenery.SVGSelfDrawable;
  
  inherit( Drawable, SVGSelfDrawable, {
    // called when the defs block changes
    updateDefs: function( defs ) {
      this.visualState.updateDefs( defs );
    },
    
    // called from elsewhere to update the SVG element
    repaint: function() {
      if ( this.dirty ) {
        this.dirty = false;
        this.visualState.updateSVG();
      }
    },
    
    dispose: function() {
      // super call
      Drawable.prototype.dispose.call( this );
      
      this.node.detachSVGDrawable( this );
    }
  } );
  
  return SVGSelfDrawable;
} );
