// Copyright 2002-2012, University of Colorado

/**
 * An Instance of a Node in the expanded tree form.
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  // layer should be null if the trail isn't to a painted node
  scenery.Instance = function( trail, layer, parent ) {
    this.trail = trail; // trail may be assumed to be stale, for performance reasons
    this.layer = layer;
    this.oldLayer = layer; // used during stitching
    
    // assertion not enabled, since at the start we don't specify a layer (it will be constructed later)
    // assert && assert( trail.lastNode().isPainted() === ( layer !== null ), 'Has a layer iff is painted' );
    
    // TODO: SVG layer might want to put data (group/fragment) references here (indexed by layer ID)
    this.data = {};
    
    // TODO: ensure that we can track this? otherwise remove it for memory and speed
    this.parent = parent;
    this.children = [];
    
    // TODO: track these? should significantly accelerate subtree-changing operations
    this.startAffectedLayer = null;
    this.endAffectedLayer = null;
    
    trail.setImmutable(); // make sure our Trail doesn't change from under us
  };
  var Instance = scenery.Instance;
  
  Instance.prototype = {
    constructor: Instance,
    
    getScene: function() {
      return this.trail.rootNode();
    },
    get scene() { return this.getScene(); },
    
    getNode: function() {
      return this.trail.lastNode();
    },
    get node() { return this.getNode(); },
    
    changeLayer: function( newLayer ) {
      if ( newLayer !== this.layer ) {
        sceneryLayerLog && sceneryLayerLog( 'changing instance ' + this.trail.toString() + ' to layer #' + newLayer.id );
        this.layer && ( this.layer._instanceCount -= 1 );
        this.layer = newLayer;
        this.layer && ( this.layer._instanceCount += 1 );
      }
    },
    
    updateLayer: function() {
      if ( this.layer !== this.oldLayer ) {
        if ( sceneryLayerLog ) {
          if ( this.oldLayer && this.layer ) {
            sceneryLayerLog( 'moving instance ' + this.trail.toString() + ' from layer #' + this.oldLayer.id + ' to layer #' + this.layer.id );
          } else if ( this.layer ) {
            sceneryLayerLog( 'adding instance ' + this.trail.toString() + ' to layer #' + this.layer.id );
          } else {
            sceneryLayerLog( 'remove instance ' + this.trail.toString() + ' from layer #' + this.oldLayer.id );
          }
        }
        if ( this.oldLayer ) {
          this.oldLayer.removeNodeFromTrail( this.trail );
        }
        if ( this.layer ) {
          this.layer.addNodeFromTrail( this.trail );
        }
      }
    },
    
    createChild: function( childNode, index ) {
      var childTrail = this.trail.copy().addDescendant( childNode );
      var childInstance = new scenery.Instance( childTrail, null, this );
      sceneryLayerLog && sceneryLayerLog( 'Instance.createChild: ' + childInstance.toString() );
      this.insertInstance( index, childInstance );
      childInstance.getNode().addInstance( childInstance );
      
      return childInstance;
    },
    
    addInstance: function( instance ) {
      assert && assert( instance, 'Instance.addInstance cannot have falsy parameter' );
      this.children.push( instance );
    },
    
    insertInstance: function( index, instance ) {
      assert && assert( instance, 'Instance.insert cannot have falsy instance parameter' );
      assert && assert( index >= 0 && index <= this.children.length, 'Instance.insert has bad index ' + index + ' for length ' + this.children.length );
      this.children.splice( index, 0, instance );
    },
    
    removeInstance: function( index ) {
      assert && assert( typeof index === 'number' );
      this.children.splice( index, 1 );
    },
    
    reindex: function() {
      this.trail.reindex();
    },
    
    // TODO: rename, so that it indicates that it removes the instance from the node
    dispose: function() {
      if ( this.layer ) {
        this.layer.removeNodeFromTrail( this.trail );
        this.layer = null;
      }
      this.parent = null;
      this.children.length = 0;
      this.getNode().removeInstance( this );
    },
    
    equals: function( other ) {
      assert && assert( ( this === other ) === this.trail.equals( other.trail ), 'We assume a 1-1 mapping from trails to instances' );
      return this === other;
    },
    
    // standard -1,0,1 comparison with another instance, as a total ordering from the render order
    compare: function( other ) {
      return this.trail.compare( other.trail );
    },
    
    getLayerString: function() {
      return this.layer ? ( this.layer.getName() + '#' + this.layer.getId() ) : '-';
    },
    
    getTrailString: function() {
      return this.trail.toString();
    },
    
    toString: function() {
      return '{' + this.getTrailString() + ', ' + this.getLayerString() + '}';
    },
    
    getAffectedLayers: function() {
      // TODO: optimize this using pre-recorded versions?
      this.reindex();
      return this.getScene().affectedLayers( this.trail );
    },
    
    /*---------------------------------------------------------------------------*
    * Events from the Node
    *----------------------------------------------------------------------------*/
    
    notifyVisibilityChange: function() {
      var thisInstance = this;
      sceneryEventLog && sceneryEventLog( 'notifyVisibilityChange: ' + this.trail.toString() + ', ' + this.getLayerString() );
      
      _.each( this.getAffectedLayers(), function( layer ) { layer.notifyVisibilityChange( thisInstance ); } );
    },
    
    notifyOpacityChange: function() {
      var thisInstance = this;
      sceneryEventLog && sceneryEventLog( 'notifyOpacityChange: ' + this.trail.toString() + ', ' + this.getLayerString() );
      
      _.each( this.getAffectedLayers(), function( layer ) { layer.notifyOpacityChange( thisInstance ); } );
    },
    
    notifyBeforeSelfChange: function() {
      sceneryEventLog && sceneryEventLog( 'notifyBeforeSelfChange: ' + this.trail.toString() + ', ' + this.getLayerString() );
      // TODO: Canvas will only need to be notified of these once in-between scene updates
      // TODO: walk up the "tree" to see if any ancestors did this (in which case we don't need to)
      // e.g. this.oldPaint = true, etc.
      this.layer.notifyBeforeSelfChange( this );
    },
    
    notifyBeforeSubtreeChange: function() {
      var thisInstance = this;
      sceneryEventLog && sceneryEventLog( 'notifyBeforeSubtreeChange: ' + this.trail.toString() + ', ' + this.getLayerString() );
      
      _.each( this.getAffectedLayers(), function( layer ) { layer.notifyBeforeSubtreeChange( thisInstance ); } );
    },
    
    notifyDirtySelfPaint: function() {
      sceneryEventLog && sceneryEventLog( 'notifyDirtySelfPaint: ' + this.trail.toString() + ', ' + this.getLayerString() );
      assert && assert( this.getNode().isPainted(), 'Instance needs to be painted for notifyDirtySelfPaint' );
      this.layer.notifyDirtySelfPaint( this );
    },
    
    // TODO: consider special post-transform type?
    notifyDirtySubtreePaint: function() {
      var thisInstance = this;
      sceneryEventLog && sceneryEventLog( 'notifyDirtySubtreePaint: ' + this.trail.toString() + ', ' + this.getLayerString() );
      
      _.each( this.getAffectedLayers(), function( layer ) { layer.notifyDirtySubtreePaint( thisInstance ); } );
    },
    
    notifyDirtySubtreeBounds: function() {
      var thisInstance = this;
      sceneryEventLog && sceneryEventLog( 'notifyDirtySubtreeBounds: ' + this.trail.toString() + ', ' + this.getLayerString() );
      
      _.each( this.getAffectedLayers(), function( layer ) { layer.notifyDirtySubtreeBounds( thisInstance ); } );
    },
    
    notifyTransformChange: function() {
      var thisInstance = this;
      sceneryEventLog && sceneryEventLog( 'notifyTransformChange: ' + this.trail.toString() + ', ' + this.getLayerString() );
      
      _.each( this.getAffectedLayers(), function( layer ) { layer.notifyTransformChange( thisInstance ); } );
    },
    
    notifyBoundsAccuracyChange: function() {
      sceneryEventLog && sceneryEventLog( 'notifyBoundsAccuracyChange: ' + this.trail.toString() + ', ' + this.getLayerString() );
      this.layer.notifyBoundsAccuracyChange( this );
    },
    
    notifyStitch: function( match ) {
      sceneryEventLog && sceneryEventLog( 'notifyStitch: ' + this.trail.toString() + ' match:' + match + ', ' + this.getLayerString() );
      this.getScene().stitch( match );
    },
    
    markForLayerRefresh: function() {
      sceneryEventLog && sceneryEventLog( 'markForLayerRefresh: ' + this.trail.toString() + ', ' + this.getLayerString() );
      this.getScene().markSceneForLayerRefresh( this );
    },
    
    markForInsertion: function( child, index ) {
      sceneryEventLog && sceneryEventLog( 'markForInsertion: ' + this.trail.toString() + ' child:' + child.id + ', index: ' + index + ', ' + this.getLayerString() );
      
      this.reindex();
      this.getScene().markSceneForInsertion( this, child, index );
    },
    
    markForRemoval: function( child, index ) {
      sceneryEventLog && sceneryEventLog( 'markForRemoval: ' + this.trail.toString() + ' child:' + child.id + ', index: ' + index + ', ' + this.getLayerString() );
      
      this.reindex();
      this.getScene().markSceneForRemoval( this, child, index );
    }
  };
  
  return Instance;
} );


