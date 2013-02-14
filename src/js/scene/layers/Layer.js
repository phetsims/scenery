// Copyright 2002-2012, University of Colorado

/**
 * Base code for layers that helps with shared layer functions
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  var Bounds2 = phet.math.Bounds2;
  
  // assumes main is wrapped with JQuery
  scenery.Layer = function( args ) {
    this.main = args.main;
    
    // initialize to fully dirty so we draw everything the first time
    // bounds in global coordinate frame
    this.dirtyBounds = Bounds2.EVERYTHING;
    
    // filled in after construction by an external source (currently Scene.rebuildLayers).
    this.startPath = null;
    this.endPath = null;
    
    // references to surrounding layers, filled by rebuildLayers
    this.nextLayer = null;
    this.previousLayer = null;
  };
  
  var Layer = scenery.Layer;
  
  Layer.prototype = {
    constructor: Layer,
    
    /*---------------------------------------------------------------------------*
    * Abstract
    *----------------------------------------------------------------------------*/
    
    // called before the layer is rendered to, with a specific render state
    initialize: function( renderState ) {
      throw new Error( 'Layer.initialize unimplemented' );
    },
    
    // called when rendering switches away from this layer
    cooldown: function( renderState ) {
      throw new Error( 'Layer.cooldown unimplemented' );
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      throw new Error( 'Layer.applyTransformationMatrix unimplemented' );
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      throw new Error( 'unimplemented layer reindex' );
    },
    
    pushClipShape: function( shape ) {
      throw new Error( 'Layer.pushClipShape unimplemented' );
    },
    
    popClipShape: function() {
      throw new Error( 'Layer.popClipShape unimplemented' );
    },
    
    // prepare a specific region for redrawing. this should clear the region with transparency if it makes sense
    prepareBounds: function( globalBounds ) {
      throw new Error( 'Layer.prepareBounds unimplemented' );
    },
    
    renderToCanvas: function( canvas, context, delayCounts ) {
      throw new Error( 'Layer.renderToCanvas unimplemented' );
    },
    
    dispose: function() {
      throw new Error( 'Layer.dispose unimplemented' );
    },
    
    /*---------------------------------------------------------------------------*
    * Implementation
    *----------------------------------------------------------------------------*/
    
    // TODO: reconsider having the dirty bounds baked into the base class!
    
    isDirty: function() {
      return !this.dirtyBounds.isEmpty();
    },
    
    markDirtyRegion: function( bounds ) {
      // TODO: for performance, consider more than just a single dirty bounding box
      this.dirtyBounds = this.dirtyBounds.union( bounds.dilated( 1 ).roundedOut() );
    },
    
    resetDirtyRegions: function() {
      this.dirtyBounds = Bounds2.NOTHING;
    },
    
    prepareDirtyRegions: function() {
      this.prepareBounds( this.dirtyBounds );
    },
    
    getDirtyBounds: function() {
      return this.dirtyBounds;
    }
    
  };
})();


