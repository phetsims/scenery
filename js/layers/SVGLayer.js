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

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Transform3 = require( 'DOT/Transform3' );
  var Matrix3 = require( 'DOT/Matrix3' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Layer = require( 'SCENERY/layers/Layer' ); // extends Layer
  require( 'SCENERY/Trail' );
  
  // used namespaces
  var svgns = 'http://www.w3.org/2000/svg';
  var xlinkns = 'http://www.w3.org/1999/xlink';
  
  scenery.SVGLayer = function( args ) {
    var $main = args.$main;
    
    // main SVG element
    this.svg = document.createElementNS( svgns, 'svg' );
    
    // the SVG has a single group under it, which corresponds to the transform of the layer's base node
    // TODO: consider renaming to 'this.baseGroup'
    this.g = document.createElementNS( svgns, 'g' );
    
    this.svg.appendChild( this.g );
    this.$svg = $( this.svg );
    this.svg.setAttribute( 'width', $main.width() );
    this.svg.setAttribute( 'height', $main.height() );
    this.svg.setAttribute( 'stroke-miterlimit', 10 ); // to match our Canvas brethren so we have the same default behavior
    this.$svg.css( 'position', 'absolute' );
    $main.append( this.svg );
    
    this.scene = args.scene;
    
    this.isSVGLayer = true;
    
    // maps trail ID => SVG self fragment (that displays shapes, text, etc.)
    this.idFragmentMap = {};
    
    // maps trail ID => SVG <g> that contains that node's self and everything under it
    this.idGroupMap = {};
    
    this.temporaryDebugFlagSoWeDontUpdateBoundariesMoreThanOnce = false;
    
    Layer.call( this, args );
    
    this.baseTransformDirty = true;
    this.baseTransformChange = true;
    
    this.initializeBoundaries();
  };
  var SVGLayer = scenery.SVGLayer;
  
  SVGLayer.prototype = _.extend( {}, Layer.prototype, {
    constructor: SVGLayer,
    
    applyTransform: function( transform, group ) {
      if ( transform.isIdentity() ) {
        if ( group.hasAttribute( 'transform' ) ) {
          group.removeAttribute( 'transform' );
        }
      } else {
        group.setAttribute( 'transform', transform.getMatrix().svgTransform() );
      }
    },
    
    // FIXME: ordering of group trees is currently not guaranteed (this just appends right now, so they need to be ensured in the proper order)
    ensureGroupTree: function( trail ) {
      if ( !( trail.getUniqueId() in this.idGroupMap ) ) {
        var subtrail = this.baseTrail.copy(); // grab the trail up to (and including) the base node, so we don't create superfluous groups
        var lastId = null;
        
        // walk a subtrail up from the root node all the way to the full trail, creating groups where necessary
        while ( subtrail.length <= trail.length ) {
          var id = subtrail.getUniqueId();
          if ( !( id in this.idGroupMap ) ) {
            if ( lastId ) {
              // we have a parent group to which we need to be added
              var group = lastId ? document.createElementNS( svgns, 'g' ) : this.g;
              this.applyTransform( subtrail.lastNode().transform, group );
              this.idGroupMap[id] = group;
              
              // TODO: handle the ordering here if we ensure group trees!
              this.idGroupMap[lastId].appendChild( group );
            } else {
              // we are ensuring the base group
              assert && assert( subtrail.lastNode() === this.baseNode );
              
              this.idGroupMap[id] = this.g;
              
              // sets up the proper transform for the base
              this.initializeBase();
            }
          }
          subtrail.addDescendant( trail.nodes[subtrail.length] );
          lastId = id;
        }
      }
    },
    
    initializeBoundaries: function() {
      if ( this.temporaryDebugFlagSoWeDontUpdateBoundariesMoreThanOnce ) {
        throw new Error( 'temporaryDebugFlagSoWeDontUpdateBoundariesMoreThanOnce!' );
      }
      this.temporaryDebugFlagSoWeDontUpdateBoundariesMoreThanOnce = true;
      
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
      var layer = this;
      if ( this.baseTransformDirty ) {
        // this will be run either now or at the end of flushing changes
        var includesBaseTransformChange = this.baseTransformChange;
        this.domChange( function() {
          layer.updateBaseTransform( includesBaseTransformChange );
        } );
        
        this.baseTransformDirty = false;
        this.baseTransformChange = false;
      }
      this.flushDOMChanges();
    },
    
    dispose: function() {
      Layer.prototype.dispose.call( this );
      this.$svg.detach();
    },
    
    markDirtyRegion: function( node, localBounds, transform, trail ) {
      // not necessary, SVG takes care of handling this (or would just redraw everything anyways)
    },
    
    markBaseTransformDirty: function( changed ) {
      var baseTransformChange = this.baseTransformChange || !!changed;
      if ( this.batchDOMChanges ) {
        this.baseTransformDirty = true;
        this.baseTransformChange = baseTransformChange;
      } else {
        this.updateBaseTransform( baseTransformChange );
      }
    },
    
    initializeBase: function() {
      // we don't want to call updateBaseTransform() twice, since baseNodeInternalBoundsChange() will call it if we use CSS transform
      if ( this.cssTransform ) {
        this.baseNodeInternalBoundsChange();
      } else {
        this.markBaseTransformDirty( true );
      }
    },
    
    // called when the base node's "internal" (self or child) bounds change, but not when it is just from the base node's own transform changing
    baseNodeInternalBoundsChange: function() {
      if ( this.cssTransform ) {
        // we want to set the baseNodeTransform to a translation so that it maps the baseNode's self/children in the baseNode's local bounds to (0,0,w,h)
        var internalBounds = this.baseNode.parentToLocalBounds( this.baseNode.getBounds() );
        var padding = scenery.Layer.cssTransformPadding;
        
        // if there is nothing, or the bounds are empty for some reason, skip this!
        if ( !internalBounds.isEmpty() ) {
          this.baseNodeTransform.set( Matrix3.translation( Math.ceil( -internalBounds.minX + padding), Math.ceil( -internalBounds.minY + padding ) ) );
          var baseNodeInteralBounds = this.baseNodeTransform.transformBounds2( internalBounds );
          
          // sanity check to ensure we are within that range
          assert && assert( baseNodeInteralBounds.minX >= 0 && baseNodeInteralBounds.minY >= 0 );
          
          this.updateContainerDimensions( Math.ceil( baseNodeInteralBounds.maxX + padding ),
                                          Math.ceil( baseNodeInteralBounds.maxY + padding ) );
        }
        
        // if this gets removed, update initializeBase()
        this.markBaseTransformDirty( true );
      } else if ( this.usesPartialCSSTransforms ) {
        this.markBaseTransformDirty( true );
      }
    },
    
    updateContainerDimensions: function( width, height ) {
      var layer = this;
      this.domChange( function() {
        layer.svg.setAttribute( 'width', width );
        layer.svg.setAttribute( 'height', height );
      } );
    },
    
    updateBaseTransform: function( includesBaseTransformChange ) {
      var transform = this.baseTrail.getTransform();
      
      if ( this.cssTransform ) {
        // set the full transform!
        this.$svg.css( transform.getMatrix().timesMatrix( this.baseNodeTransform.getInverse() ).cssTransformStyles() );
        
        if ( includesBaseTransformChange ) {
          this.applyTransform( this.baseNodeTransform, this.g );
        }
      } else if ( this.usesPartialCSSTransforms ) {
        // calculate what our CSS transform should be
        var cssTransform = new Transform3();
        var matrix = transform.getMatrix();
        if ( this.cssTranslation ) {
          cssTransform.append( Matrix3.translation( matrix.m02(), matrix.m12() ) );
        }
        if ( this.cssRotation ) {
          cssTransform.append( Matrix3.rotation2( matrix.rotation() ) );
        }
        if ( this.cssScale ) {
          var scaling = matrix.scaling();
          cssTransform.append( Matrix3.scaling( scaling.x, scaling.y ) );
        }
        
        // take the CSS transform out of what we will apply to the group
        transform.prepend( cssTransform.getInverse() );
        
        // now we need to see where our baseNode bounds are mapped to with our transform,
        // so that we can apply an extra translation and adjust dimensions as necessary
        var padding = scenery.Layer.cssTransformPadding;
        var internalBounds = this.baseNode.parentToLocalBounds( this.baseNode.getBounds() );
        var mappedBounds = transform.transformBounds2( internalBounds );
        var translation = Matrix3.translation( Math.ceil( -mappedBounds.minX + padding ), Math.ceil( -mappedBounds.minY + padding ) );
        var inverseTranslation = translation.inverted();
        this.updateContainerDimensions( Math.ceil( mappedBounds.width()  + 2 * padding ),
                                        Math.ceil( mappedBounds.height() + 2 * padding ) );
        
        // put the translation adjustment and its inverse in-between the two transforms
        cssTransform.append( inverseTranslation );
        transform.prepend( translation );
        
        // apply the transforms
        // TODO: checks to make sure we don't apply them in a row if one didn't change!
        this.$svg.css( cssTransform.getMatrix().cssTransformStyles() );
        this.applyTransform( transform, this.g );
      } else {
        this.applyTransform( transform, this.g );
      }
    },
    
    transformChange: function( args ) {
      var layer = this;
      var node = args.node;
      var trail = args.trail;
      
      if ( trail.lastNode() === this.baseNode ) {
        // our trail points to the base node. handle this case as special
        this.markBaseTransformDirty();
      } else if ( _.contains( trail.nodes, this.baseNode ) ) {
        var group = this.idGroupMap[trail.getUniqueId()];
        
        // apply the transform to the group
        this.domChange( function() {
          layer.applyTransform( node.transform, group );
        } );
      } else {
        // ancestor node changed a transform. rebuild the base transform
        this.markBaseTransformDirty();
      }
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
    
    getSVGString: function() {
      // TODO: jQuery seems to be stripping namespaces, so figure that one out?
      return $( '<div>' ).append( this.$svg.clone() ).html();
      
      // also note:
      // var doc = document.implementation.createHTMLDocument("");
      // doc.write(html);
       
      // // You must manually set the xmlns if you intend to immediately serialize the HTML
      // // document to a string as opposed to appending it to a <foreignObject> in the DOM
      // doc.documentElement.setAttribute("xmlns", doc.documentElement.namespaceURI);
       
      // // Get well-formed markup
      // html = (new XMLSerializer).serializeToString(doc);
    },
    
    // TODO: note for DOM we can do https://developer.mozilla.org/en-US/docs/HTML/Canvas/Drawing_DOM_objects_into_a_canvas
    renderToCanvas: function( canvas, context, delayCounts ) {
      // temporarily put the full transform on the containing group so the rendering is correct (CSS transforms can take this away)
      this.applyTransform( this.baseTrail.getTransform(), this.g );
      
      if ( window.canvg ) {
        delayCounts.increment();
        
        // TODO: if we are using CSS3 transforms, run that here
        canvg( canvas, this.getSVGString(), {
          ignoreMouse: true,
          ignoreAnimation: true,
          ignoreDimensions: true,
          ignoreClear: true,
          renderCallback: function() {
            delayCounts.decrement();
          }
        } );
      } else {
        // will not work on Internet Explorer 9/10
        
        // TODO: very much not convinced that this is better than setting src of image
        var DOMURL = window.URL || window.webkitURL || window;
        var img = new Image();
        var raw = this.getSVGString();
        console.log( raw );
        var svg = new Blob( [ raw ] , { type: "image/svg+xml;charset=utf-8" } );
        var url = DOMURL.createObjectURL( svg );
        delayCounts.increment();
        img.onload = function() {
          context.drawImage( img, 0, 0 );
          // TODO: this loading is delayed!!! ... figure out a solution to potentially delay?
          DOMURL.revokeObjectURL( url );
          delayCounts.decrement();
        };
        img.src = url;
        
        throw new Error( 'this implementation hits Chrome bugs, won\'t work on IE9/10, etc. deprecated' );
      }
      
      // revert the transform damage that we did to our base group
      this.updateBaseTransform();
    },
    
    getName: function() {
      return 'svg';
    }
  } );
  
  return SVGLayer;
} );


