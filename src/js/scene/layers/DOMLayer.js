// Copyright 2002-2012, University of Colorado

/**
 * A DOM-based layer in the scene graph. Each layer handles dirty-region handling separately,
 * and corresponds to a single canvas / svg element / DOM element in the main container.
 * Importantly, it does not contain rendered content from a subtree of the main
 * scene graph. It only will render a contiguous block of nodes visited in a depth-first
 * manner.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  var Bounds2 = phet.math.Bounds2;
  
  scenery.DOMLayer = function( args ) {
    scenery.Layer.call( this, args );
    
    this.div = document.createElement( 'div' );
    this.$div = $( this.div );
    this.$div.width( this.$main.width() );
    this.$div.height( this.$main.height() );
    this.$div.css( 'position', 'absolute' );
    this.$main.append( this.div );
    
    this.scene = args.scene;
    
    this.isDOMLayer = true;
  };
  
  var DOMLayer = scenery.DOMLayer;
  
  DOMLayer.prototype = _.extend( {}, scenery.Layer.prototype, {
    constructor: DOMLayer,
    
    updateBoundaries: function( entry ) {
      scenery.Layer.prototype.updateBoundaries.call( this, entry );
      
      var layer = this;
      
      // TODO: assumes that nodes under this are in a tree, not a DAG
      this.startPointer.eachNodeBetween( this.endPointer, function( node ) {
        // all nodes should have DOM support if node.hasSelf()
        if ( node.hasSelf() ) {
          node.addToDOMLayer( layer );
        }
      } );
    },
    
    render: function( scene, args ) {
      // nothing at all needed here, CSS transforms taken care of when dirty regions are notified
    },
    
    dispose: function() {
      this.$div.detach();
    },
    
    markDirtyRegion: function( args ) {
      // no-op for now, since it is not scenery's job to change DOM elements. transforms are handled with transformChange.
    },
    
    transformChange: function( args ) {
      var trail = args.trail;
      var baseTransform = args.transform;
      
      // TODO: iterate only over nodes in this layer (sub-trees may not be contained!)
      new scenery.TrailPointer( trail, true ).eachNodeBetween( new scenery.TrailPointer( trail, false ), function( node ) {
        if ( node.hasSelf() ) {
          // TODO: mark that this is for DOM nodes?
          node.updateCSSTransform( baseTransform );
        }
      } );
      throw new Error( 'FIXME: improper handling of the transform, since we are setting the transform to all children!' );
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      // nothing at all needed here
    },
    
    getContainer: function() {
      return this.div;
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      this.$div.css( 'z-index', zIndex );
      this.zIndex = zIndex;
      return zIndex + 1;
    },
    
    pushClipShape: function( shape ) {
      // TODO: clipping
    },
    
    popClipShape: function() {
      // TODO: clipping
    },
    
    // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
    renderToCanvas: function( canvas, context, delayCounts ) {
      // TODO: consider not silently failing?
      // var data = "<svg xmlns='http://www.w3.org/2000/svg' width='" + this.$main.width() + "' height='" + this.$main.height() + "'>" +
      //   "<foreignObject width='100%' height='100%'>" +
      //   $( this.div ).html() +
      //   "</foreignObject></svg>";
      
      // var DOMURL = window.URL || window.webkitURL || window;
      // var img = new Image();
      // var svg = new Blob( [ data ] , { type: "image/svg+xml;charset=utf-8" } );
      // var url = DOMURL.createObjectURL( svg );
      // delayCounts.increment();
      // img.onload = function() {
      //   context.drawImage( img, 0, 0 );
      //   // TODO: this loading is delayed!!! ... figure out a solution to potentially delay?
      //   DOMURL.revokeObjectURL( url );
      //   delayCounts.decrement();
      // };
      // img.src = url;
    },
    
    getName: function() {
      return 'dom';
    }
  } );
})();


