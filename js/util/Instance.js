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
  require( 'SCENERY/util/AccessibilityPeer' );
  
  // layer should be null if the trail isn't to a painted node
  scenery.Instance = function( trail, layer ) {
    this.trail = trail; // trail may be assumed to be stale, for performance reasons
    this.layer = layer;
    
    assert && assert( trail.lastNode().isPainted() === ( layer !== null ), 'Has a layer iff is painted' );
    
    // TODO: SVG layer might want to put data (group/fragment) references here (indexed by layer ID)
    this.data = {};
    
    // TODO: ensure that we can track this? otherwise remove it for memory and speed
    this.parent = null;
    this.children = [];
    
    this.peers = []; // list of AccessibilityPeer instances attached to this trail
    
    // TODO: track these? should significantly accelerate subtree-changing operations
    this.startAffectedLayer = null;
    this.endAffectedLayer = null;
    
    trail.setImmutable(); // make sure our Trail doesn't change from under us
    
    this.addPeers();
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
      this.layer = newLayer;
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
    
    dispose: function() {
      this.parent = null;
      this.children.length = 0;
      this.getNode().removeInstance( this );
      
      this.removePeers();
    },
    
    equals: function( other ) {
      assert && assert( ( this === other ) === this.trail.equals( other.trail ), 'We assume a 1-1 mapping from trails to instances' );
      return this === other;
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
    
    addPeers: function() {
      var thisInstance = this;
      var node = this.getNode();
      var scene = this.getScene();
      
      if ( node._peers.length ) {
        _.each( node._peers, function( desc ) {
          var peer = new scenery.AccessibilityPeer( thisInstance, desc.element, desc.options );
          scene.addPeer( peer );
          thisInstance.peers.push( peer );
        } );
      }
    },
    
    removePeers: function() {
      var scene = this.getScene();
      
      _.each( this.peers, function( peer ) {
        scene.removePeer( peer );
        peer.dispose();
      } );
      
      this.peers.length = 0; // clear this.peers
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


