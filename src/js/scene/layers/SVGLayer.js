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
  
  var svgns = 'http://www.w3.org/2000/svg';
  var xlinkns = 'http://www.w3.org/1999/xlink';
  
  scenery.SVGLayer = function( args ) {
    scenery.Layer.call( this, args );
    
    this.svg = document.createElementNS( svgns, 'svg' );
    this.g = document.createElementNS( svgns, 'g' );
    this.$svg = $( this.svg );
    this.svg.setAttribute( 'width', this.$main.width() );
    this.svg.setAttribute( 'height', this.$main.height() );
    this.svg.setAttribute( 'stroke-miterlimit', 10 );
    this.$svg.css( 'position', 'absolute' );
    this.$main.append( this.svg );
    
    this.scene = args.scene;
    
    this.isSVGLayer = true;
  };
  
  var SVGLayer = scenery.SVGLayer;
  
  SVGLayer.prototype = _.extend( {}, scenery.Layer.prototype, {
    constructor: SVGLayer,
    
    applyGroup: function( node, group ) {
      if ( node.transform.isIdentity() ) {
        if ( group.hasAttribute( 'transform' ) ) {
          group.removeAttribute( 'transform' );
        }
      } else {
        group.setAttribute( 'transform', node.transform.getMatrix().svgTransform() );
      }
    },
    
    updateBoundaries: function( entry ) {
      scenery.Layer.prototype.updateBoundaries.call( this, entry );
      
      var layer = this;
      
      // TODO: assumes that nodes under this are in a tree, not a DAG
      this.startPointer.eachNodeBetween( this.endPointer, function( node ) {
        // all nodes should have DOM support if node.hasSelf()
        if ( node.hasSelf() ) {
          var svgFragment = node.createSVGFragment();
          //node.addToSVGLayer( layer );
        }
      } );
    },
    
    render: function( scene, args ) {
      // nothing at all needed here, CSS transforms taken care of when dirty regions are notified
    },
    
    dispose: function() {
      this.$svg.detach();
    },
    
    markDirtyRegion: function( node, localBounds, transform, trail ) {
      // for now, update the transforms for the node and any children in the layer that it may have
      // TODO: should we catch a separate event, transform-change?
      // new scenery.TrailPointer( trail, true ).eachNodeBetween( new scenery.TrailPointer( trail, false ), function( node ) {
      //   if ( node.hasSelf() ) {
      //     node.updateCSSTransform( transform );
      //   }
      // } );
    },
    
    // TODO: consider a stack-based model for transforms?
    applyTransformationMatrix: function( matrix ) {
      // nothing at all needed here
    },
    
    getContainer: function() {
      return this.svg;
    },
    
    // returns next zIndex in place. allows layers to take up more than one single zIndex
    reindex: function( zIndex ) {
      this.$svg.css( 'z-index', zIndex );
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
      // TODO: consider canvg?
      throw new Error( 'SVGLayer.renderToCanvas unimplemented' );
    },
    
    getName: function() {
      return 'svg';
    }
  } );
})();


