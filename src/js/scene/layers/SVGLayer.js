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
    this.svg = document.createElementNS( svgns, 'svg' );
    this.g = document.createElementNS( svgns, 'g' );
    this.svg.appendChild( this.g );
    this.$svg = $( this.svg );
    this.svg.setAttribute( 'width', this.$main.width() );
    this.svg.setAttribute( 'height', this.$main.height() );
    this.svg.setAttribute( 'stroke-miterlimit', 10 ); // to match our Canvas brethren so we have the same default behavior
    this.$svg.css( 'position', 'absolute' );
    this.$main.append( this.svg );
    
    this.scene = args.scene;
    
    this.isSVGLayer = true;
    
    // maps trail ID => SVG self fragment (that displays shapes, text, etc.)
    this.idFragmentMap = {};
    
    // maps trail ID => SVG <g> that contains that node's self and everything under it
    this.idGroupMap = {};
    
    this.temporaryDebugFlagSoWeDontUpdateBoundariesMoreThanOnce = false;
    
    scenery.Layer.call( this, args );
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
    
    // FIXME: ordering of group trees is currently not guaranteed (this just appends right now, so they need to be ensured in the proper order)
    ensureGroupTree: function( trail ) {
      if ( !( trail.getUniqueId() in this.idGroupMap ) ) {
        var subtrail = new scenery.Trail( trail.rootNode() );
        var lastId = null;
        
        // walk a subtrail up from the root node all the way to the full trail, creating groups where necessary
        while ( subtrail.length <= trail.length ) {
          var id = subtrail.getUniqueId();
          if ( !( id in this.idGroupMap ) ) {
            var group = document.createElementNS( svgns, 'g' );
            this.applyGroup( subtrail.lastNode(), group );
            this.idGroupMap[id] = group;
            if ( lastId ) {
              // we have a parent group to which we need to be added
              // TODO: handle the ordering here if we ensure group trees!
              this.idGroupMap[lastId].appendChild( group );
            } else {
              // no parent, so append ourselves to the SVGLayer's master group
              this.g.appendChild( group );
            }
          }
          subtrail.addDescendant( trail.nodes[subtrail.length] );
          lastId = id;
        }
      }
    },
    
    updateBoundaries: function( entry ) {
      if ( this.temporaryDebugFlagSoWeDontUpdateBoundariesMoreThanOnce ) {
        throw new Error( 'temporaryDebugFlagSoWeDontUpdateBoundariesMoreThanOnce!' );
      }
      this.temporaryDebugFlagSoWeDontUpdateBoundariesMoreThanOnce = true;
      
      scenery.Layer.prototype.updateBoundaries.call( this, entry );
      
      var layer = this;
      
      // TODO: consider removing SVG fragments from our dictionary? if we burn through a lot of one-time fragments we will memory leak like crazy
      // TODO: handle updates. insertion is helpful based on the trail, as we can find where to insert nodes
      
      this.startPointer.eachTrailBetween( this.endPointer, function( trail ) {
        var node = trail.lastNode();
        var trailId = trail.getUniqueId();
        
        layer.ensureGroupTree( trail );
        
        if ( node.hasSelf() ) {
          var svgFragment = node.createSVGFragment();
          layer.idFragmentMap[trailId] = svgFragment;
          layer.idGroupMap[trailId].appendChild( svgFragment );
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
      // not necessary, SVG takes care of handling this (or would just redraw everything anyways)
    },
    
    transformChange: function( args ) {
      var node = args.node;
      var trail = args.trail;
      
      // TODO: find the associated group!
      var group;
      
      // apply the transform to the group
      this.applyGroup( node, group );
      
      throw new Error( 'group lookup not implemented' );
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


