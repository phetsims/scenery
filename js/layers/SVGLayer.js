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
  require( 'SCENERY/util/Trail' );
  
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
    
    // the <defs> block that we will be stuffing gradients and patterns into
    this.defs = document.createElementNS( svgns, 'defs' );
    
    var width = $main.width();
    var height = $main.height();
    
    this.svg.appendChild( this.defs );
    this.svg.appendChild( this.g );
    this.$svg = $( this.svg );
    this.svg.setAttribute( 'width', width );
    this.svg.setAttribute( 'height', height );
    this.svg.setAttribute( 'stroke-miterlimit', 10 ); // to match our Canvas brethren so we have the same default behavior
    this.$svg.css( 'position', 'absolute' );
    this.svg.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
    this.svg.style['pointer-events'] = 'none';
    $main.append( this.svg );
    
    this.scene = args.scene;
    
    this.isSVGLayer = true;
    
    // maps trail ID => SVG self fragment (that displays shapes, text, etc.)
    this.idFragmentMap = {};
    
    // maps trail ID => SVG <g> that contains that node's self and everything under it
    this.idGroupMap = {};
    
    
    Layer.call( this, args );
    
    this.baseTransformDirty = true;
    this.baseTransformChange = true;
    
    this.initializeBoundaries();
  };
  var SVGLayer = scenery.SVGLayer;
  
  SVGLayer.prototype = _.extend( {}, Layer.prototype, {
    constructor: SVGLayer,
    
    /*
     * Notes about how state is tracked here:
     * Trails are stored on group.trail so that we can look this up when inserting new groups
     */
    addNodeFromTrail: function( trail ) {
      assert && assert( !( trail.getUniqueId() in this.idFragmentMap ), 'Already contained that trail!' );
      assert && assert( trail.lastNode().hasSelf(), 'Don\'t add nodes without hasSelf() to SVGLayer' );
      
      var subtrail = this.baseTrail.copy(); // grab the trail up to (and including) the base node, so we don't create superfluous groups
      var lastId = null;
      
      // walk a subtrail up from the root node all the way to the full trail, creating groups where necessary
      while ( subtrail.length <= trail.length ) {
        var id = subtrail.getUniqueId();
        if ( !( id in this.idGroupMap ) ) {
          // we need to create a new group
          var group;
          
          if ( lastId ) {
            // we have a parent group to which we need to be added
            group = document.createElementNS( svgns, 'g' );
            
            // apply the node's transform to the group
            this.applyTransform( subtrail.lastNode().getTransform(), group );
            
            // add the group to its parent
            this.insertGroupIntoParent( group, this.idGroupMap[lastId], subtrail );
          } else {
            // we are ensuring the base group
            assert && assert( subtrail.lastNode() === this.baseNode );
            
            group = this.g;
            
            // sets up the proper transform for the base
            this.initializeBase();
          }
          
          group.referenceCount = 0; // initialize a reference count, so we can know when to remove unused groups
          group.trail = subtrail.copy(); // put a reference to the trail on the group, so we can efficiently scan and see where to insert future groups
          
          this.idGroupMap[id] = group;
        }
        
        // this trail will depend on this group, so increment the reference counter
        this.idGroupMap[id].referenceCount++;
        
        // step down towards our full trail
        subtrail.addDescendant( trail.nodes[subtrail.length] );
        lastId = id;
      }
      
      // actually add the node into its own group
      var node = trail.lastNode();
      var trailId = trail.getUniqueId();
      
      var nodeGroup = this.idGroupMap[trailId];
      var svgFragment = node.createSVGFragment( this.svg, this.defs, nodeGroup );
      this.updateNode( node, svgFragment );
      this.updateNodeGroup( node, nodeGroup );
      this.idFragmentMap[trailId] = svgFragment;
      nodeGroup.appendChild( svgFragment );
    },
    
    removeNodeFromTrail: function( trail ) {
      assert && assert( !( trail.getUniqueId() in this.idFragmentMap ), 'Already contained that trail!' );
      
      // clean up the fragment and defs directly died to the node
      var trailId = trail.getUniqueId();
      var node = trail.lastNode();
      var fragment = this.idFragmentMap[trailId];
      this.idGroupMap[trailId].removeChild( fragment );
      delete this.idFragmentMap[trailId];
      if ( node.removeSVGDefs ) {
        node.removeSVGDefs( this.svg, this.defs );
      }
      
      // clean up any unneeded groups
      var subtrail = trail.copy();
      while ( subtrail.length > this.baseTrail.length ) {
        var id = subtrail.getUniqueId();
        
        var group = this.idGroupMap[id];
        group.referenceCount--;
        if ( group.referenceCount === 0 ) {
          // completely kill the group
          group.parentNode.removeChild( group );
          delete group.trail; // just in case someone held a reference
          delete this.idGroupMap[id];
        }
        
        subtrail.removeDescendant();
      }
      this.g.referenceCount--; // since we don't go down to the base group, adjust its reference count
    },
    
    // subtrail is to group, and should include parentGroup below
    insertGroupIntoParent: function( group, parentGroup, subtrail ) {
      if ( !parentGroup.childNodes.length ) {
        parentGroup.appendChild( group );
      } else {
        // if there is already a child, we need to do a scan to ensure we place our group as a child in the correct order (above/below)
        
        // scan other child groups in the parentGroup to find where we need to be (index i)
        var afterNode = null;
        var indexIndex = subtrail.length - 2; // index into the trail's indices
        var ourIndex = subtrail.indices[indexIndex];
        var i;
        for ( i = 0; i < parentGroup.childNodes.length; i++ ) {
          var child = parentGroup.childNodes[i];
          if ( child.trail ) {
            child.trail.reindex();
            var otherIndex = child.trail.indices[indexIndex];
            if ( otherIndex > ourIndex ) {
              // this other group is above us
              break;
            }
          }
        }
        
        // insert our group before parentGroup.childNodes[i] (or append if that doesn't exist)
        if ( i === parentGroup.childNodes.length ) {
          parentGroup.appendChild( group );
        } else {
          parentGroup.insertBefore( group, parentGroup.childNodes[i] );
        }
      }
    },
    
    // updates visual styles on an existing SVG fragment
    updateNode: function( node, fragment ) {
      if ( node.updateSVGFragment ) {
        node.updateSVGFragment( fragment );
      }
      if ( node.updateSVGDefs ) {
        node.updateSVGDefs( this.svg, this.defs );
      }
    },
    
    // updates necessary paint attributes on a group (not including transform)
    updateNodeGroup: function( node, group ) {
      if ( node.isVisible() ) {
        group.style.display = 'inherit';
      } else {
        group.style.display = 'none';
      }
    },
    
    applyTransform: function( transform, group ) {
      if ( transform.isIdentity() ) {
        if ( group.hasAttribute( 'transform' ) ) {
          group.removeAttribute( 'transform' );
        }
      } else {
        group.setAttribute( 'transform', transform.getMatrix().getSVGTransform() );
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
              this.applyTransform( subtrail.lastNode().getTransform(), group );
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
      var layer = this;
      
      // TODO: consider removing SVG fragments from our dictionary? if we burn through a lot of one-time fragments we will memory leak like crazy
      // TODO: handle updates. insertion is helpful based on the trail, as we can find where to insert nodes
      
      this.startPointer.eachTrailBetween( this.endPointer, function( trail ) {
        var node = trail.lastNode();
        var trailId = trail.getUniqueId();
        
        layer.ensureGroupTree( trail );
        
        if ( node.hasSelf() ) {
          var group = layer.idGroupMap[trailId];
          var svgFragment = node.createSVGFragment( layer.svg, layer.defs, group );
          layer.updateNode( node, svgFragment );
          layer.updateNodeGroup( node, group );
          layer.idFragmentMap[trailId] = svgFragment;
          group.appendChild( svgFragment );
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
    
    markDirtyRegion: function( args ) {
      var node = args.node;
      var trailId = args.trail.getUniqueId();
      
      var fragment = this.idFragmentMap[trailId];
      if ( fragment ) {
        this.updateNode( node, fragment );
      }
      
      var group = this.idGroupMap[trailId];
      if ( group ) {
        this.updateNodeGroup( node, group );
      }
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
        this.$svg.css( transform.getMatrix().timesMatrix( this.baseNodeTransform.getInverse() ).getCSSTransformStyles() );
        
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
          cssTransform.append( Matrix3.rotation2( matrix.getRotation() ) );
        }
        if ( this.cssScale ) {
          var scaleVector = matrix.getScaleVector();
          cssTransform.append( Matrix3.scaling( scaleVector.x, scaleVector.y ) );
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
        this.updateContainerDimensions( Math.ceil( mappedBounds.getWidth()  + 2 * padding ),
                                        Math.ceil( mappedBounds.getHeight() + 2 * padding ) );
        
        // put the translation adjustment and its inverse in-between the two transforms
        cssTransform.append( inverseTranslation );
        transform.prepend( translation );
        
        // apply the transforms
        // TODO: checks to make sure we don't apply them in a row if one didn't change!
        this.$svg.css( cssTransform.getMatrix().getCSSTransformStyles() );
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
          layer.applyTransform( node.getTransform(), group );
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


