// Copyright 2002-2013, University of Colorado

/**
 * Main scene, that is also a Node.
 *
 * TODO: documentation!
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  'use strict';
  
  var collect = require( 'PHET_CORE/collect' );
  var inherit = require( 'PHET_CORE/inherit' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );
  var Matrix3 = require( 'DOT/Matrix3' );
  
  var Shape = require( 'KITE/Shape' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // inherits from Node
  require( 'SCENERY/util/Instance' );
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/RenderInterval' );
  require( 'SCENERY/util/TrailPointer' );
  require( 'SCENERY/input/Input' );
  require( 'SCENERY/layers/LayerBuilder' );
  require( 'SCENERY/layers/Renderer' );
  
  var Util = require( 'SCENERY/util/Util' );
  var objectCreate = Util.objectCreate;
  
  var accessibility = window.has && window.has( 'scenery.accessibility' );
  
  // debug flag to disable matching of layers when in 'match' mode
  var forceNewLayers = true; // DEBUG
  
  /*
   * $main should be a block-level element with a defined width and height. scene.resize() should be called whenever
   * it is resized.
   *
   * Valid parameters in the parameter object:
   * {
   *   allowSceneOverflow: false,           // usually anything displayed outside of this $main (DOM/CSS3 transformed SVG) is hidden with CSS overflow
   *   allowCSSHacks: true,                 // applies styling that prevents mobile browser graphical issues
   *   allowDevicePixelRatioScaling: false, // allows underlying canvases (Canvas, WebGL) to increase in size to maintain sharpness on high-density displays
   *   enablePointerEvents: true,           // allows pointer events / MSPointerEvent to be used on supported platforms.
   *   preferredSceneLayerType: ...,        // sets the preferred type of layer to be created if there are multiple options
   *   width: <current main width>,         // override the main container's width
   *   height: <current main height>,       // override the main container's height
   * }
   */
  scenery.Scene = function Scene( $main, options ) {
    sceneryAssert && sceneryAssert( $main[0], 'A main container is required for a scene' );
    this.$main = $main;
    this.main = $main[0];
    
    // add a self reference to aid in debugging. this generally shouldn't lead to a memory leak
    this.main.scene = this;
    
    // add a reference to the API for debugging
    this.scenery = scenery;
    
    // defaults
    options = _.extend( {
      allowSceneOverflow: false,
      allowCSSHacks: true,
      allowDevicePixelRatioScaling: false,
      enablePointerEvents: true,
      preferredSceneLayerType: scenery.CanvasDefaultLayerType,
      width: $main.width(),
      height: $main.height()
    }, options || {} );
    
    // TODO: consider using a pushed preferred layer to indicate this information, instead of as a specific option
    this.backingScale = options.allowDevicePixelRatioScaling ? Util.backingScale( document.createElement( 'canvas' ).getContext( '2d' ) ) : 1;
    this.enablePointerEvents = options.enablePointerEvents;
    
    Node.call( this, options );
    
    var scene = this;
    window.debugScene = scene;
    
    // layering data
    this.layers = [];               // main layers in a scene
    this.layerChangeIntervals = []; // array of {RenderInterval}s indicating what parts need to be stitched together. cleared after each stitching
    
    this.lastCursor = null;
    this.defaultCursor = $main.css( 'cursor' );
    
    // resize the main container as a sanity check
    this.setSize( options.width, options.height );
    
    this.sceneBounds = new Bounds2( 0, 0, options.width, options.height );
    
    // set up the root instance for this scene
    // only do this after Node.call has been invoked, since Trail.addDescendant uses a few things
    this.rootInstance = new scenery.Instance( new scenery.Trail( this ), null, null );
    this.addInstance( this.rootInstance );
    
    // default to a canvas layer type, but this can be changed
    this.preferredSceneLayerType = options.preferredSceneLayerType;
    
    applyCSSHacks( $main, options );
    
    if ( accessibility ) {
      this.activePeer = null;
      
      this.accessibilityLayer = document.createElement( 'div' );
      this.accessibilityLayer.className = "accessibility-layer";
      
      //Put the accessibility layer behind the background so it cannot be seen.  Change this to some high number like 9999 to show it for debugging purposes.
      this.accessibilityLayer.style.zIndex = -1;
      this.accessibilityLayer.style.position = 'relative';
      $main[0].appendChild( this.accessibilityLayer );
      
      this.focusRingSVGContainer = document.createElementNS( 'http://www.w3.org/2000/svg', 'svg' );
      this.focusRingSVGContainer.style.position = 'absolute';
      this.focusRingSVGContainer.style.top = 0;
      this.focusRingSVGContainer.style.left = 0;
      this.focusRingSVGContainer.style['pointer-events'] = 'none';
      this.resizeFocusRingSVGContainer( options.width, options.height );
      this.focusRingPath = document.createElementNS( 'http://www.w3.org/2000/svg', 'path' );
      this.focusRingPath.setAttribute( 'style', 'fill: none; stroke: blue; stroke-width: 5;' );
      this.focusRingPath.setAttribute( 'id', 'p1' );
      this.focusRingSVGContainer.appendChild( this.focusRingPath );
      $main[0].appendChild( this.focusRingSVGContainer );
      
      this.updateFocusRing = function() {
        // TODO: move into prototype definitions, this doesn't need to be private, and isn't a closure over anything in the constructor
        sceneryAssert && sceneryAssert( scene.activePeer, 'scene should have an active peer when changing the focus ring bounds' );
        scene.focusRingPath.setAttribute( 'd', Shape.bounds( scene.activePeer.getGlobalBounds() ).getSVGPath() );
      };

      //Put the live region layer behind the accessibility peer layer to make debugging easier (if we need to see the accessibility layer)
      this.liveRegionLayer = document.createElement( 'div' );
      this.liveRegionLayer.className = 'live-region-layer';
      this.liveRegionLayer.style.zIndex = -2;
      this.liveRegionLayer.style.position = 'relative';
      $main[0].appendChild( this.liveRegionLayer );
    }
  };
  var Scene = scenery.Scene;
  
  inherit( Node, Scene, {
    
    updateScene: function( args ) {
      // sceneryLayerLog && sceneryLayerLog( 'Scene: updateScene' );
      
      var scene = this;
      
      // validating bounds, similar to Piccolo2d
      this.validateBounds();
      this.validatePaint();
      
      // bail if there are no layers. consider a warning?
      if ( !this.layers.length ) {
        return;
      }
      
      _.each( this.layers, function( layer ) {
        layer.render( scene, args );
      } );
      
      this.updateCursor();
      
      // if ( this.accessibilityLayer ) {
  //      for ( var i = 0; i < accessibleNodes.length; i++ ) {
  //        if ( accessibleNodes[i]._element === activeElement ) {
  //          if ( accessibleNodes[i].origin ) {
  //            var b = accessibleNodes[i].origin.globalBounds;
  //            var rect = Shape.bounds( b );
  //
  //            //Animation is a bit buggy, but I left this code in in case we want to pick it up later.
  //            var animateTheRect = false;
  //            if ( animateTheRect ) {
  //              if ( !this.focusRingPath.lastSVGPath ) {
  //                this.focusRingPath.setAttribute( 'd', rect.getSVGPath() );
  //                this.focusRingPath.lastSVGPath = rect.getSVGPath();
  //              } else {
  //                var animate = document.createElementNS( 'http://www.w3.org/2000/svg', 'animate' );
  //                animate.setAttribute( 'attributeType', 'XML' );
  //                animate.setAttribute( 'xlink:href', '#p1' );
  //                animate.setAttribute( 'attributeName', 'd' );
  //                animate.setAttribute( 'from', this.focusRingPath.lastSVGPath );
  //                animate.setAttribute( 'to', rect.getSVGPath() );
  //                animate.setAttribute( 'dur', '4s' );
  //
  //                $( this.focusRingPath ).empty();
  //                this.focusRingPath.appendChild( animate );
  //                this.focusRingPath.lastSVGPath = rect.getSVGPath();
  //              }
  //            } else {
  //              this.focusRingPath.setAttribute( 'd', rect.getSVGPath() );
  //            }
  //
  //            found = true;
  //          }
  //        }
  //        if ( !found ) {
  //          this.focusRingPath.removeAttribute( 'd' );
  //        }
  //      }
      // }
    },
    
    renderScene: function() {
      // TODO: for now, go with the same path. possibly add options later
      this.updateScene();
    },
    
    addPeer: function( peer ) {
      this.accessibilityLayer.appendChild( peer.peerElement );
    },
    
    removePeer: function( peer ) {
      this.accessibilityLayer.removeChild( peer.peerElement );
    },

    addLiveRegion: function( liveRegion ) {
      this.liveRegionLayer.appendChild( liveRegion.element );
    },

    removeLiveRegion: function( liveRegion ) {
      this.liveRegionLayer.removeChild( liveRegion.element );
    },
    
    setActivePeer: function( peer ) {
      if ( this.activePeer !== peer ) {
        var scene = this;
        
        //Remove bounds listener from old active peer
        if ( this.activePeer ) {
          this.activePeer.instance.node.removeEventListener( 'bounds', this.updateFocusRing );
        }
        
        this.activePeer = peer;
        
        if ( peer ) {
          this.activePeer.instance.node.addEventListener( 'bounds', this.updateFocusRing );
          this.updateFocusRing();
        } else {
          this.focusRingPath.setAttribute( 'd', "M 0 0" );
        }
      }
    },
    
    getActivePeer: function( peer ) {
      return this.activePeer;
    },
    
    focusPeer: function( peer ) {
      this.setActivePeer( peer );
    },
    
    blurPeer: function( peer ) {
      sceneryAssert && sceneryAssert( this.getActivePeer() === peer, 'Can only blur an active peer' );
      this.setActivePeer( null );
    },
    
    markInterval: function( affectedTrail ) {
      // TODO: maybe reindexing sooner is better? are we covering up a bug here?
      affectedTrail.reindex();
      
      // since this is marked while the child is still connected, we can use our normal trail handling.
      
      // find the closest before and after self trails that are not affected
      var beforeTrail = affectedTrail.previousPainted(); // easy for the before trail
      
      var afterTrailPointer = new scenery.TrailPointer( affectedTrail.copy(), false );
      while ( afterTrailPointer.hasTrail() && ( !afterTrailPointer.isBefore || !afterTrailPointer.trail.isPainted() ) ) {
        afterTrailPointer.nestedForwards();
      }
      var afterTrail = afterTrailPointer.trail;
      
      // sanity checks
      sceneryAssert && sceneryAssert( !beforeTrail || beforeTrail.areIndicesValid(), 'beforeTrail needs to be valid' );
      sceneryAssert && sceneryAssert( !afterTrail || afterTrail.areIndicesValid(), 'afterTrail needs to be valid' );
      sceneryAssert && sceneryAssert( !beforeTrail || !afterTrail || beforeTrail.compare( afterTrail ) !== 0, 'Marked interval needs to be exclusive' );
      
      // store the layer of the before/after trails so that it is easy to access later
      this.addLayerChangeInterval( new scenery.RenderInterval( beforeTrail, afterTrail ) );
    },
    
    // convenience function for layer change intervals
    addLayerChangeInterval: function( interval ) {
      if ( sceneryLayerLog ) {
        sceneryLayerLog( 'adding interval: ' + interval.toString() + ' to intervals:' );
        _.each( this.layerChangeIntervals, function( interval ) {
          sceneryLayerLog( '  ' + interval.toString() );
        } );
      }
      
      // TODO: replace with a binary-search-like version that may be faster. this includes a full scan
      
      // attempt to merge this interval with another if possible.
      for ( var i = 0; i < this.layerChangeIntervals.length; i++ ) {
        var other = this.layerChangeIntervals[i];
        other.reindex(); // sanity check, although for most use-cases this should be unnecessary
        
        if ( interval.exclusiveUnionable( other ) ) {
          // the interval can be unioned without including other nodes. do this, and remove the other interval from consideration
          interval = interval.union( other );
          this.layerChangeIntervals.splice( i--, 1 ); // decrement to stay at the same index
          sceneryLayerLog && sceneryLayerLog( 'removing interval: ' + other.toString() );
        }
      }
      
      this.layerChangeIntervals.push( interval );
      
      if ( sceneryLayerLog ) {
        sceneryLayerLog( 'new intervals: ' );
        _.each( this.layerChangeIntervals, function( interval ) {
          sceneryLayerLog( '  ' + interval.toString() );
        } );
        sceneryLayerLog( '---' );
      }
    },
    
    createLayer: function( layerType, layerArgs, startBoundary, endBoundary ) {
      var layer = layerType.createLayer( _.extend( {
        startBoundary: startBoundary,
        endBoundary: endBoundary
      }, layerArgs ) );
      layer.type = layerType;
      sceneryLayerLog && sceneryLayerLog( 'created layer: ' + layer.getId() + ' of type ' + layer.type.name );
      return layer;
    },
    
    // insert a layer into the proper place (from its starting boundary)
    insertLayer: function( layer ) {
      for ( var i = 0; i < this.layers.length; i++ ) {
        if ( layer.endPaintedTrail.isBefore( this.layers[i].startPaintedTrail ) ) {
          this.layers.splice( i, 0, layer ); // insert the layer here
          return;
        }
      }
      
      // it is after all other layers
      this.layers.push( layer );
    },
    
    getBoundaries: function() {
      // TODO: store these more efficiently!
      return [ this.layers[0].startBoundary ].concat( _.pluck( this.layers, 'endBoundary' ) );
    },
    
    calculateBoundaries: function( beforeLayerType, beforeTrail, afterTrail ) {
      sceneryLayerLog && sceneryLayerLog( 'build between ' + ( beforeTrail ? beforeTrail.toString() : beforeTrail ) + ',' + ( afterTrail ? afterTrail.toString() : afterTrail ) + ' with beforeType: ' + ( beforeLayerType ? beforeLayerType.name : null ) );
      var builder = new scenery.LayerBuilder( this, beforeLayerType, beforeTrail, afterTrail );
      
      // push the preferred layer type before we push that for any nodes
      if ( this.preferredSceneLayerType ) {
        builder.pushPreferredLayerType( this.preferredSceneLayerType );
      }
      
      builder.run();
      
      return builder.boundaries;
    },
    
    stitch: function( match ) {
      var scene = this;
      
      sceneryLayerLog && sceneryLayerLog( '-----------------------------------\nbeginning stitch' );
      
      // bail out if there are no changes to stitch (stitch is called multiple times)
      if ( !this.layerChangeIntervals.length ) {
        return;
      }
      
      // data to be shared across all of the individually stitched intervals
      var stitchData = {
        // all instances that are affected, in no particular order (and may contain duplicates)
        affectedInstances: [],
        
        // fresh layers that should be added into the scene
        newLayers: []
      };
      
      // default arguments for constructing layers
      var layerArgs = {
        $main: this.$main,
        scene: this,
        baseNode: this
      };
      
      _.each( this.layerChangeIntervals, function( interval ) {
        // reindex intervals, since their endpoints indices may need to be updated
        interval.reindex();
      } );
      
      /*
       * Sort our intervals, so that when we need to 'unglue' a layer into two separate layers, we will have passed
       * all of the parts where we would need to use the 'before' layer, so we can update our layer map with the 'after'
       * layer.
       */
      this.layerChangeIntervals.sort( scenery.RenderInterval.compareDisjoint );
      
      sceneryLayerLog && sceneryLayerLog( 'stitching on intervals: \n' + this.layerChangeIntervals.join( '\n' ) );
      
      _.each( this.layerChangeIntervals, function( interval ) {
        sceneryLayerLog && sceneryLayerLog( 'stitch on interval ' + interval.toString() );
        var beforeTrail = interval.start;
        var afterTrail = interval.end;
        
        var beforeInstance = beforeTrail ? beforeTrail.getInstance() : null;
        var afterInstance = afterTrail ? afterTrail.getInstance() : null;
        
        var beforeLayer = beforeInstance ? beforeInstance.layer : null;
        var afterLayer = afterInstance ? afterInstance.layer : null;
        
        // TODO: calculate boundaries based on the instances?
        var boundaries = scene.calculateBoundaries( beforeLayer ? beforeLayer.type : null, beforeTrail, afterTrail );
        
        scene.stitchInterval( stitchData, layerArgs, beforeTrail, afterTrail, beforeLayer, afterLayer, boundaries, match );
      } );
      
      // clean up state that was set leading up to the stitching, and do it early so
      // if we do things later that cause side-effects we won't clear intervals that haven't been stitched
      this.layerChangeIntervals.length = 0;
      
      sceneryLayerLog && sceneryLayerLog( '------ finished intervals in stitching' );
      
      // reindex all of the relevant layer trails
      _.each( this.layers.concat( stitchData.newLayers ), function( layer ) {
        layer.startBoundary.reindex();
        layer.endBoundary.reindex(); // TODO: performance: this repeats some work, verify in layer audit that we are sharing boundaries properly, then only reindex end boundary on last layer
      } );
      
      // remove necessary layers. do this before adding layers, since insertLayer currently does not gracefully handle weird overlapping cases
      _.each( this.layers.slice( 0 ), function( layer ) {
        // layers with zero trails should be removed
        if ( layer._instanceCount === 0 ) {
          sceneryLayerLog && sceneryLayerLog( 'disposing layer: ' + layer.getId() );
          scene.disposeLayer( layer );
        }
      } );
      
      // add new layers. we do this before the add/remove trails, since those can trigger layer side effects
      _.each( stitchData.newLayers, function( layer ) {
        sceneryAssert && sceneryAssert( layer._instanceCount, 'ensure we are not adding empty layers' );
        
        sceneryLayerLog && sceneryLayerLog( 'inserting layer: ' + layer.getId() );
        scene.insertLayer( layer );
      } );
      
      // set the layers' elements' z-indices, and reindex their trails so they are in a consistent state
      // TODO: performance: don't reindex layers if no layers were added or removed?
      this.reindexLayers();
      
      sceneryLayerLog && sceneryLayerLog( '------ updating layer references' );
      
      // add/remove trails from their necessary layers
      _.each( stitchData.affectedInstances, function( instance ) {
        instance.updateLayer();
      } );
      
      sceneryAssertExtra && sceneryAssertExtra( this.layerAudit() );
      
      sceneryLayerLog && sceneryLayerLog( 'finished stitch\n-----------------------------------' );
    },
    
    /*
     * Stitching intervals has essentially two specific modes:
     * non-matching: handles added or removed nodes (and this can span multiple, even adjacent trails)
     * matching: handles in-place layer refreshes (no nodes removed or added, but something like a renderer was changed)
     *
     * This separation occurs since for matching, we want to match old layers with possible new layers, so we can keep trails in their
     * current layer instead of creating an identical layer and moving the trails to that layer.
     *
     * The stitching basically re-does the layering between a start and end trail, attempting to minimize the amount of changes made.
     * It can include 'gluing' layers together (a node that caused layer splits was removed, and before/after layers are joined),
     * 'ungluing' layers (an inserted node causes a layer split in an existing layer, and it is separated into a before/after),
     * or normal updating of the interior.
     *
     * The beforeTrail and afterTrail should be outside the modifications, and if the modifications are to the start/end of the graph,
     * they should be passed as null to indicate 'before everything' and 'after everything' respectively.
     *
     * Here be dragons!
     */
    stitchInterval: function( stitchData, layerArgs, beforeTrail, afterTrail, beforeLayer, afterLayer, boundaries, match ) {
      var scene = this;
      
      // make sure our beforeTrail and afterTrail are immutable
      beforeTrail && beforeTrail.setImmutable();
      afterTrail && afterTrail.setImmutable();
      
      // need a reference to this, since it may change
      var afterLayerEndBoundary = afterLayer ? afterLayer.endBoundary : null;
      
      var beforeLayerIndex = beforeLayer ? _.indexOf( this.layers, beforeLayer ) : -1;
      var afterLayerIndex = afterLayer ? _.indexOf( this.layers, afterLayer ) : this.layers.length;
      
      var beforePointer = beforeTrail ? new scenery.TrailPointer( beforeTrail, true ) : new scenery.TrailPointer( new scenery.Trail( this ), true );
      var afterPointer = afterTrail ? new scenery.TrailPointer( afterTrail, true ) : new scenery.TrailPointer( new scenery.Trail( this ), false );
      
      sceneryLayerLog && sceneryLayerLog( '\nstitching with boundaries:\n' + _.map( boundaries, function( boundary ) { return boundary.toString(); } ).join( '\n' ) );
      sceneryLayerLog && sceneryLayerLog( '               layers: ' + ( beforeLayer ? beforeLayer.getId() : '-' ) + ' to ' + ( afterLayer ? afterLayer.getId() : '-' ) );
      sceneryLayerLog && sceneryLayerLog( '               trails: ' + ( beforeTrail ? beforeTrail.toString() : '-' ) + ' to ' + ( afterTrail ? afterTrail.toString() : '-' ) );
      sceneryLayerLog && sceneryLayerLog( '               match: ' + match );
      
      /*---------------------------------------------------------------------------*
      * State
      *----------------------------------------------------------------------------*/
      
      var nextBoundaryIndex = 0;
      var nextBoundary = boundaries[nextBoundaryIndex];
      var instancesToAddToLayer = [];
      var currentTrail = beforeTrail;
      var currentLayer = beforeLayer;
      var currentLayerType = beforeLayer ? beforeLayer.type : null;
      var currentStartBoundary = null;
      var matchingLayer = null; // set whenever a trail has a matching layer, cleared after boundary
      
      function addPendingTrailsToLayer() {
        // add the necessary nodes to the layer
        _.each( instancesToAddToLayer, function( instance ) {
          instance.changeLayer( currentLayer );
          stitchData.affectedInstances.push( instance );
        } );
        instancesToAddToLayer.length = 0;
      }
      
      function addAndCreateLayer( startBoundary, endBoundary ) {
        currentLayer = scene.createLayer( currentLayerType, layerArgs, startBoundary, endBoundary );
        stitchData.newLayers.push( currentLayer );
      }
      
      function step( trail, isEnd ) {
        sceneryLayerLog && sceneryLayerLog( 'step: ' + ( trail ? trail.toString() : trail ) );
        trail && trail.setImmutable(); // we don't want our trail to be modified, so we can store direct references to it
        // check for a boundary at this step between currentTrail and trail
        
        // if there is no next boundary, don't bother checking anyways
        if ( nextBoundary && nextBoundary.equivalentPreviousTrail( currentTrail ) ) { // at least one null check
          sceneryAssert && sceneryAssert( nextBoundary.equivalentNextTrail( trail ) );
          
          sceneryLayerLog && sceneryLayerLog( nextBoundary.toString() );
          
          // we are at a boundary change. verify that we are at the end of a layer
          if ( currentLayer || currentStartBoundary ) {
            if ( currentLayer ) {
              sceneryLayerLog && sceneryLayerLog( 'has currentLayer: ' + currentLayer.getId() );
              // existing layer, reposition its endpoint
              currentLayer.setEndBoundary( nextBoundary );
            } else {
              sceneryAssert && sceneryAssert( currentStartBoundary );
              
              if ( matchingLayer ) {
                sceneryLayerLog && sceneryLayerLog( 'matching layer used: ' + matchingLayer.getId() );
                matchingLayer.setStartBoundary( currentStartBoundary );
                matchingLayer.setEndBoundary( nextBoundary );
                currentLayer = matchingLayer;
              } else {
                sceneryLayerLog && sceneryLayerLog( 'creating layer' );
                addAndCreateLayer( currentStartBoundary, nextBoundary ); // sets currentLayer
              }
            }
            // sanity checks
            sceneryAssert && sceneryAssert( currentLayer.startPaintedTrail );
            sceneryAssert && sceneryAssert( currentLayer.endPaintedTrail );
            
            addPendingTrailsToLayer();
          } else {
            // if not at the end of a layer, sanity check that we should have no accumulated pending trails
            sceneryLayerLog && sceneryLayerLog( 'was first layer' );
            sceneryAssert && sceneryAssert( instancesToAddToLayer.length === 0 );
          }
          currentLayer = null;
          currentLayerType = nextBoundary.nextLayerType;
          currentStartBoundary = nextBoundary;
          matchingLayer = null;
          nextBoundaryIndex++;
          nextBoundary = boundaries[nextBoundaryIndex];
        }
        if ( trail && !isEnd ) {
          // TODO: performance: handle instances natively, don't just convert here
          instancesToAddToLayer.push( trail.getInstance() );
        }
        if ( match && !isEnd ) { // TODO: verify this condition with test cases
          // if the node's old layer is compatible
          // TODO: performance: don't use getInstance() here, use instances natively
          var layer = trail.getInstance().layer; // lookup should return the old layer from the system
          if ( layer.type === currentLayerType && !forceNewLayers ) {
            // TODO: we need to handle compatibility with layer splits. using forceNewLayers flag to temporarily disable
            matchingLayer = layer;
          }
        }
        currentTrail = trail;
      }
      
      function startStep( trail ) {
        sceneryLayerLog && sceneryLayerLog( 'startStep: ' + ( trail ? trail.toString() : trail ) );
      }
      
      function middleStep( trail ) {
        sceneryLayerLog && sceneryLayerLog( 'middleStep: ' + trail.toString() );
        step( trail, false );
      }
      
      function endStep( trail ) {
        sceneryLayerLog && sceneryLayerLog( 'endStep: ' + ( trail ? trail.toString() : trail ) );
        step( trail, true );
        
        if ( beforeLayer !== afterLayer && boundaries.length === 0 ) {
          // glue the layers together
          sceneryLayerLog && sceneryLayerLog( 'gluing layer' );
          sceneryLayerLog && sceneryLayerLog( 'endBoundary: ' + afterLayer.endBoundary.toString() );
          beforeLayer.setEndBoundary( afterLayer.endBoundary );
          currentLayer = beforeLayer;
          addPendingTrailsToLayer();
          
          // move over all of afterLayer's trails to beforeLayer
          // defensive copy needed, since this will be modified at the same time
          _.each( afterLayer._layerTrails.slice( 0 ), function( trail ) {
            trail.reindex();
            var instance = trail.getInstance();
            instance.changeLayer( beforeLayer ); // TODO: performance: handle instances natively
            stitchData.affectedInstances.push( instance );
          } );
          
        } else if ( beforeLayer && beforeLayer === afterLayer && boundaries.length > 0 ) {
          // need to 'unglue' and split the layer
          sceneryLayerLog && sceneryLayerLog( 'ungluing layer' );
          sceneryAssert && sceneryAssert( currentStartBoundary );
          addAndCreateLayer( currentStartBoundary, afterLayerEndBoundary ); // sets currentLayer
          addPendingTrailsToLayer();
          
          currentLayer.endPaintedTrail.reindex(); // currentLayer's trails may be stale at this point
          scenery.Trail.eachPaintedTrailBetween( afterTrail, currentLayer.endPaintedTrail, function( subtrail ) {
            var instance = subtrail.getInstance();
            instance.changeLayer( currentLayer );
            stitchData.affectedInstances.push( instance );
          }, false, scene );
        } else if ( !beforeLayer && !afterLayer && boundaries.length === 1 && !boundaries[0].hasNext() && !boundaries[0].hasPrevious() ) {
          // TODO: why are we generating a boundary here?!?
        } else {
          currentLayer = afterLayer;
          // TODO: check concepts on this guard, since it seems sketchy
          if ( currentLayer && currentStartBoundary ) {
            currentLayer.setStartBoundary( currentStartBoundary );
          }
          
          addPendingTrailsToLayer();
        }
      }
      
      // iterate from beforeTrail up to BEFORE the afterTrail. does not include afterTrail
      startStep( beforeTrail );
      beforePointer.eachTrailBetween( afterPointer, function( trail ) {
        // ignore non-self trails
        if ( !trail.isPainted() || ( beforeTrail && trail.equals( beforeTrail ) ) ) {
          return;
        }
        
        middleStep( trail.copy() );
      } );
      endStep( afterTrail );
    },
    
    rebuildLayers: function() {
      sceneryLayerLog && sceneryLayerLog( 'Scene: rebuildLayers' );
      
      // mark the entire scene 
      this.markInterval( new scenery.Trail( this ) );
      
      // then stitch with match=true
      this.stitch( true );
    },
    
    // after layer changes, the layers should have their zIndex updated, and updates their trails
    reindexLayers: function() {
      sceneryLayerLog && sceneryLayerLog( 'Scene: reindexLayers' );
      
      var index = 1; // don't start below 1
      if ( accessibility && this.accessibiltyLayer ) {
        this.accessibilityLayer.style.zIndex = 9999; // TODO: a better way than 9999, SR says probably unnecessary
        index++;
      }
      
      _.each( this.layers, function( layer ) {
        // layers increment indices as needed
        index = layer.reindex( index );
      } );
      
      if ( accessibility ) {
        if ( this.focusRingSVGContainer ) {
          this.focusRingSVGContainer.style.zIndex = index;
        }
      }
    },
    
    dispose: function() {
      this.disposeLayers();
      if ( this.input ) {
        this.input.disposeListeners();
      }
      
      // remove self reference from the container
      delete this.main.scene;
      
      // TODO: clear event handlers if added
      //throw new Error( 'unimplemented dispose: clear event handlers if added' );
    },
    
    disposeLayer: function( layer ) {
      layer.dispose();
      this.layers.splice( _.indexOf( this.layers, layer ), 1 ); // TODO: better removal code!
    },
    
    disposeLayers: function() {
      var scene = this;
      
      _.each( this.layers.slice( 0 ), function( layer ) {
        scene.disposeLayer( layer );
      } );
    },
    
    // all layers whose start or end points lie inclusively in the range from the trail's before and after
    affectedLayers: function( trail ) {
      // midpoint search and result depends on the order of layers being in render order (bottom to top)
      
      sceneryAssert && sceneryAssert( !( trail.isEmpty() || trail.nodes[0] !== this ), 'layerLookup root matches' );
      
      var n = this.layers.length;
      if ( n === 0 ) {
        sceneryAssert && sceneryAssert( !trail.lastNode().isPainted(), 'There should be at least one layer for a painted trail' );
        return [];
      }
      
      sceneryAssert && sceneryAssert( trail.areIndicesValid() );
      
      // point to the beginning of the node, right before it would be rendered
      var startPointer = new scenery.TrailPointer( trail, true );
      var endPointer = new scenery.TrailPointer( trail, false );
      
      var layers = this.layers;
      
      // from layers 0 to n-1, notAfter goes from false to true, notBefore goes from true to false
      var low = -1;
      var high = n;
      var mid;
      
      // midpoint search to see where our trail's start isn't after a layer's end
      while ( high - 1 > low ) {
        mid = ( high + low ) >> 1;
        var endTrail = layers[mid].endPaintedTrail;
        sceneryAssert && sceneryAssert( endTrail.areIndicesValid() );
        // NOTE TO SELF: don't change this flag to true again. think it through
        var notAfter = startPointer.compareNested( new scenery.TrailPointer( endTrail, true ) ) !== 1;
        if ( notAfter ) {
          high = mid;
        } else {
          low = mid;
        }
      }
      
      // store result and reset bound
      var firstIndex = high;
      low = -1;
      high = n;
      
      // midpoint search to see where our trail's end isn't before a layer's start
      while ( high - 1 > low ) {
        mid = ( high + low ) >> 1;
        var startTrail = layers[mid].startPaintedTrail;
        startTrail.reindex();
        sceneryAssert && sceneryAssert( startTrail.areIndicesValid() );
        var notBefore = endPointer.compareNested( new scenery.TrailPointer( startTrail, true ) ) !== -1;
        if ( notBefore ) {
          low = mid;
        } else {
          high = mid;
        }
      }
      
      var lastIndex = low;
      
      return layers.slice( firstIndex, lastIndex + 1 );
    },
    
    // attempt to render everything currently visible in the scene to an external canvas. allows copying from canvas layers straight to the other canvas
    renderToCanvas: function( canvas, context, callback ) {
      var count = 0;
      var started = false; // flag guards against asynchronous tests that call back synchronously (immediate increment and decrement)
      var delayCounts = {
        increment: function() {
          count++;
        },
        
        decrement: function() {
          count--;
          if ( count === 0 && callback && started ) {
            callback();
          }
        }
      };
      
      context.clearRect( 0, 0, canvas.width, canvas.height );
      _.each( this.layers, function( layer ) {
        layer.renderToCanvas( canvas, context, delayCounts );
      } );
      
      if ( count === 0 ) {
        // no asynchronous layers, callback immediately
        if ( callback ) {
          callback();
        }
      } else {
        started = true;
      }
    },
    
    // TODO: consider SVG data URLs
    canvasDataURL: function( callback ) {
      this.canvasSnapshot( function( canvas ) {
        callback( canvas.toDataURL() );
      } );
    },
    
    // renders what it can into a Canvas (so far, Canvas and SVG layers work fine)
    canvasSnapshot: function( callback ) {
      var canvas = document.createElement( 'canvas' );
      canvas.width = this.sceneBounds.getWidth();
      canvas.height = this.sceneBounds.getHeight();
      
      var context = canvas.getContext( '2d' );
      this.renderToCanvas( canvas, context, function() {
        callback( canvas, context.getImageData( 0, 0, canvas.width, canvas.height ) );
      } );
    },
    
    // TODO: Note that this is private, better name?
    setSize: function( width, height ) {
      // resize our main container
      this.$main.width( width );
      this.$main.height( height );
      
      // set the container's clipping so anything outside won't show up
      // TODO: verify this clipping doesn't reduce performance!
      this.$main.css( 'clip', 'rect(0px,' + width + 'px,' + height + 'px,0px)' );
      
      this.sceneBounds = new Bounds2( 0, 0, width, height );
    },
    
    resize: function( width, height ) {
      this.setSize( width, height );
      this.rebuildLayers(); // TODO: why? - change this to resize individual layers
      
      if ( accessibility ) {
        this.resizeAccessibilityLayer( width, height );
        this.resizeFocusRingSVGContainer( width, height );
        
        //Update the focus ring when the scene resizes.  Note: as of 5/10/2013 this only works properly when scaling up, and is buggy (off by a translation) when scaling down
        if ( this.updateFocusRing && this.activePeer) {
          // this.updateScene();
          this.updateFocusRing();
        }
      }
      
      this.fireEvent( 'resize', { width: width, height: height } );
    },
    
    resizeAccessibilityLayer: function( width, height ) {
      if ( this.accessibilityLayer ) {
        this.accessibilityLayer.setAttribute( 'width', width );
        this.accessibilityLayer.setAttribute( 'height', height );
        this.accessibilityLayer.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
      }
    },
    
    resizeFocusRingSVGContainer: function( width, height ) {
      if ( this.focusRingSVGContainer ) {
        this.focusRingSVGContainer.setAttribute( 'width', width );
        this.focusRingSVGContainer.setAttribute( 'height', height );
        this.focusRingSVGContainer.style.clip = 'rect(0px,' + width + 'px,' + height + 'px,0px)';
      }
    },
    
    getSceneWidth: function() {
      return this.sceneBounds.getWidth();
    },
    
    getSceneHeight: function() {
      return this.sceneBounds.getHeight();
    },
    
    markSceneForLayerRefresh: function( instance ) {
      sceneryLayerLog && sceneryLayerLog( 'Scene: marking layer refresh: ' + instance.trail.toString() );
      this.markInterval( instance.trail );
    },
    
    markSceneForInsertion: function( instance, child, index ) {
      var affectedTrail = instance.trail.copy().addDescendant( child );
      sceneryLayerLog && sceneryLayerLog( 'Scene: marking insertion: ' + affectedTrail.toString() );
      
      sceneryLayerLog && sceneryLayerLog( 'inserting instances onto ' + instance.toString() + ' with child ' + child.id + ' and index ' + index );
      var baseInstance = instance.createChild( child, index );
      
      // constructs all sub-trees for the specified instance
      function buildInstances( instance ) {
        _.each( instance.getNode().children, function( child, index ) {
          var nextInstance = instance.createChild( child, index );
          buildInstances( nextInstance );
        } );
      }
      buildInstances( baseInstance );
      
      this.markInterval( affectedTrail );
    },
    
    markSceneForRemoval: function( instance, child, index ) {
      // mark the interval
      var affectedTrail = instance.trail.copy().addDescendant( child );
      sceneryLayerLog && sceneryLayerLog( 'Scene: marking removal: ' + affectedTrail.toString() );
      this.markInterval( affectedTrail );
      
      // remove the necessary instances
      var toRemove = [ instance.children[index] ];
      instance.removeInstance( index );
      while ( toRemove.length ) {
        var item = toRemove.pop();
        sceneryAssert && sceneryAssert( item, 'item instance should always exist' );
        
        // add its children
        Array.prototype.push.apply( toRemove, item.children );
        
        item.dispose(); // removes it from the node and sets it up for easy GC
      }
    },
    
    updateCursor: function() {
      if ( this.input && this.input.mouse.point ) {
        var mouseTrail = this.trailUnderPoint( this.input.mouse.point, { isMouse: true } );
        
        if ( mouseTrail ) {
          for ( var i = mouseTrail.length - 1; i >= 0; i-- ) {
            var cursor = mouseTrail.nodes[i].getCursor();
            
            if ( cursor ) {
              this.setSceneCursor( cursor );
              return;
            }
          }
        }
      }
      
      // fallback case
      this.setSceneCursor( this.defaultCursor );
    },
    
    setSceneCursor: function( cursor ) {
      if ( cursor !== this.lastCursor ) {
        this.lastCursor = cursor;
        this.$main.css( 'cursor', cursor );
      }
    },
    
    updateOnRequestAnimationFrame: function( element ) {
      var scene = this;
      (function step() {
        window.requestAnimationFrame( step, element );
        scene.updateScene();
      })();
    },
    
    initializeStandaloneEvents: function( parameters ) {
      // TODO extract similarity between standalone and fullscreen!
      var element = this.$main[0];
      this.initializeEvents( _.extend( {}, {
        listenerTarget: element,
        pointFromEvent: function( evt ) {
          var mainBounds = element.getBoundingClientRect();
          return new Vector2( evt.clientX - mainBounds.left, evt.clientY - mainBounds.top );
        }
      }, parameters ) );
    },
    
    initializeFullscreenEvents: function( parameters ) {
      var element = this.$main[0];
      this.initializeEvents( _.extend( {}, {
        listenerTarget: document,
        pointFromEvent: function( evt ) {
          var mainBounds = element.getBoundingClientRect();
          return new Vector2( evt.clientX - mainBounds.left, evt.clientY - mainBounds.top );
        }
      }, parameters ) );
    },
    
    initializeEvents: function( parameters ) {
      var scene = this;
      
      if ( scene.input ) {
        throw new Error( 'Attempt to attach events twice to the scene' );
      }
      
      // TODO: come up with more parameter names that have the same string length, so it looks creepier
      var pointFromEvent = parameters.pointFromEvent;
      var listenerTarget = parameters.listenerTarget;
      var batchDOMEvents = parameters.batchDOMEvents;
      
      var input = new scenery.Input( scene, listenerTarget, !!batchDOMEvents );
      scene.input = input;
      
      // maps the current MS pointer types onto the pointer spec
      function msPointerType( evt ) {
        if ( evt.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_TOUCH ) {
          return 'touch';
        } else if ( evt.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_PEN ) {
          return 'pen';
        } else if ( evt.pointerType === window.MSPointerEvent.MSPOINTER_TYPE_MOUSE ) {
          return 'mouse';
        } else {
          return evt.pointerType; // hope for the best
        }
      }
      
      function forEachChangedTouch( evt, callback ) {
        for ( var i = 0; i < evt.changedTouches.length; i++ ) {
          // according to spec (http://www.w3.org/TR/touch-events/), this is not an Array, but a TouchList
          var touch = evt.changedTouches.item( i );
          
          callback( touch.identifier, pointFromEvent( touch ) );
        }
      }
      
      // TODO: massive boilerplate reduction! closures should help tons!
      
      var implementsPointerEvents = window.navigator && window.navigator.pointerEnabled; // W3C spec for pointer events
      var implementsMSPointerEvents = window.navigator && window.navigator.msPointerEnabled; // MS spec for pointer event
      if ( this.enablePointerEvents && implementsPointerEvents ) {
        // accepts pointer events corresponding to the spec at http://www.w3.org/TR/pointerevents/
        input.addListener( 'pointerdown', function( domEvent ) {
          input.pointerDown( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'pointerup', function( domEvent ) {
          input.pointerUp( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'pointermove', function( domEvent ) {
          input.pointerMove( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'pointerover', function( domEvent ) {
          input.pointerOver( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'pointerout', function( domEvent ) {
          input.pointerOut( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'pointercancel', function( domEvent ) {
          input.pointerCancel( domEvent.pointerId, domEvent.pointerType, pointFromEvent( domEvent ), domEvent );
        } );
      } else if ( this.enablePointerEvents && implementsMSPointerEvents ) {
        input.addListener( 'MSPointerDown', function( domEvent ) {
          input.pointerDown( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'MSPointerUp', function( domEvent ) {
          input.pointerUp( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'MSPointerMove', function( domEvent ) {
          input.pointerMove( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'MSPointerOver', function( domEvent ) {
          input.pointerOver( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'MSPointerOut', function( domEvent ) {
          input.pointerOut( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'MSPointerCancel', function( domEvent ) {
          input.pointerCancel( domEvent.pointerId, msPointerType( domEvent ), pointFromEvent( domEvent ), domEvent );
        } );
      } else {
        input.addListener( 'mousedown', function( domEvent ) {
          input.mouseDown( pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'mouseup', function( domEvent ) {
          input.mouseUp( pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'mousemove', function( domEvent ) {
          input.mouseMove( pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'mouseover', function( domEvent ) {
          input.mouseOver( pointFromEvent( domEvent ), domEvent );
        } );
        input.addListener( 'mouseout', function( domEvent ) {
          input.mouseOut( pointFromEvent( domEvent ), domEvent );
        } );
        
        input.addListener( 'touchstart', function( domEvent ) {
          forEachChangedTouch( domEvent, function( id, point ) {
            input.touchStart( id, point, domEvent );
          } );
        } );
        input.addListener( 'touchend', function( domEvent ) {
          forEachChangedTouch( domEvent, function( id, point ) {
            input.touchEnd( id, point, domEvent );
          } );
        } );
        input.addListener( 'touchmove', function( domEvent ) {
          forEachChangedTouch( domEvent, function( id, point ) {
            input.touchMove( id, point, domEvent );
          } );
        } );
        input.addListener( 'touchcancel', function( domEvent ) {
          forEachChangedTouch( domEvent, function( id, point ) {
            input.touchCancel( id, point, domEvent );
          } );
        } );
      }
      
      input.addListener( 'keyup', function( domEvent ) {
        input.keyUp( domEvent );
      } );
      input.addListener( 'keydown', function( domEvent ) {
        input.keyDown( domEvent );
      } );
      input.addListener( 'keypress', function( domEvent ) {
        input.keyPress( domEvent );
      } );
    },
    
    getTrailFromKeyboardFocus: function() {
      // return the root (scene) trail by default
      // TODO: fill in with actual keyboard focus
      return new scenery.Trail( this );
    },
    
    fireBatchedEvents: function() {
      this.input.fireBatchedEvents();
    },
    
    resizeOnWindowResize: function() {
      var scene = this;
      
      var resizer = function() {
        scene.resize( window.innerWidth, window.innerHeight );
      };
      $( window ).resize( resizer );
      resizer();
    },
    
    // in-depth check to make sure everything is layered properly
    layerAudit: function() {
      var scene = this;
      
      var boundaries = this.calculateBoundaries( null, null, null );
      sceneryAssert && sceneryAssert( boundaries.length === this.layers.length + 1, 'boundary count (' + boundaries.length + ') does not match layer count (' + this.layers.length + ') + 1' );
      
      // count how many 'self' trails there are
      var eachTrailUnderPaintedCount = 0;
      new scenery.Trail( this ).eachTrailUnder( function( trail ) {
        if ( trail.isPainted() ) {
          eachTrailUnderPaintedCount++;
          
          sceneryAssert && sceneryAssert( trail.getInstance(), 'every painted trail must have an instance' );
        }
        
        sceneryAssert && sceneryAssert( trail.getInstance() && trail.getInstance().trail.equals( trail ), 'every trail must have a single corresponding instance' );
      } );
      
      var layerPaintedCount = 0;
      _.each( this.layers, function( layer ) {
        layerPaintedCount += layer.getLayerTrails().length;
        
        // reindex now so we don't have problems later
        layer.startPaintedTrail.reindex();
        layer.endPaintedTrail.reindex();
      } );
      
      var layerIterationPaintedCount = 0;
      _.each( this.layers, function( layer ) {
        var selfCount = 0;
        scenery.Trail.eachPaintedTrailBetween( layer.startPaintedTrail, layer.endPaintedTrail, function( trail ) {
          selfCount++;
        }, false, scene );
        sceneryAssert && sceneryAssert( selfCount > 0, 'every layer must have at least one self trail' );
        layerIterationPaintedCount += selfCount;
      } );
      
      sceneryAssert && sceneryAssert( eachTrailUnderPaintedCount === layerPaintedCount, 'cross-referencing self trail counts: layerPaintedCount, ' + eachTrailUnderPaintedCount + ' vs ' + layerPaintedCount );
      sceneryAssert && sceneryAssert( eachTrailUnderPaintedCount === layerIterationPaintedCount, 'cross-referencing self trail counts: layerIterationPaintedCount, ' + eachTrailUnderPaintedCount + ' vs ' + layerIterationPaintedCount );
      
      _.each( this.layers, function( layer ) {
        sceneryAssert && sceneryAssert( layer.startPaintedTrail.compare( layer.endPaintedTrail ) <= 0, 'proper ordering on layer trails' );
      } );
      
      for ( var i = 1; i < this.layers.length; i++ ) {
        sceneryAssert && sceneryAssert( this.layers[i-1].endPaintedTrail.compare( this.layers[i].startPaintedTrail ) === -1, 'proper ordering of layer trail boundaries in scene.layers array' );
        sceneryAssert && sceneryAssert( this.layers[i-1].endBoundary === this.layers[i].startBoundary, 'proper sharing of boundaries' );
      }
      
      _.each( this.layers, function( layer ) {
        // a list of trails that the layer tracks
        var layerTrails = layer.getLayerTrails();
        
        // a list of trails that the layer should be tracking (between painted trails)
        var computedTrails = [];
        scenery.Trail.eachPaintedTrailBetween( layer.startPaintedTrail, layer.endPaintedTrail, function( trail ) {
          computedTrails.push( trail.copy() );
        }, false, scene );
        
        // verify that the layer has an identical record of trails compared to the trails inside its boundaries
        sceneryAssert && sceneryAssert( layerTrails.length === computedTrails.length, 'layer has incorrect number of tracked trails' );
        _.each( layerTrails, function( trail ) {
          sceneryAssert && sceneryAssert( _.some( computedTrails, function( otherTrail ) { return trail.equals( otherTrail ); } ), 'layer has a tracked trail discrepancy' );
        } );
        
        // verify that each trail has the same (or null) renderer as the layer
        scenery.Trail.eachTrailBetween( layer.startPaintedTrail, layer.endPaintedTrail, function( trail ) {
          var node = trail.lastNode();
          sceneryAssert && sceneryAssert( !node.renderer || node.renderer.name === layer.type.name, 'specified renderers should match the layer renderer' );
        }, false, scene );
      } );
      
      // verify layer splits
      new scenery.Trail( this ).eachTrailUnder( function( trail ) {
        var beforeSplitTrail;
        var afterSplitTrail;
        if ( trail.lastNode().layerSplitBefore ) {
          beforeSplitTrail = trail.previousPainted();
          afterSplitTrail = trail.lastNode().isPainted() ? trail : trail.nextPainted();
          sceneryAssert && sceneryAssert( !beforeSplitTrail || !afterSplitTrail || beforeSplitTrail.getInstance().layer !== afterSplitTrail.getInstance().layer, 'layerSplitBefore layers need to be different' );
        }
        if ( trail.lastNode().layerSplitAfter ) {
          // shift a pointer from the (nested) end of the trail to the next isBefore (if available)
          var ptr = new scenery.TrailPointer( trail.copy(), false );
          while ( ptr && ptr.isAfter ) {
            ptr = ptr.nestedForwards();
          }
          
          // if !ptr, we walked off the end of the graph (nothing after layer split, automatically ok)
          if ( ptr ) {
            beforeSplitTrail = ptr.trail.previousPainted();
            afterSplitTrail = ptr.trail.lastNode().isPainted() ? ptr.trail : ptr.trail.nextPainted();
            sceneryAssert && sceneryAssert( !beforeSplitTrail || !afterSplitTrail || beforeSplitTrail.getInstance().layer !== afterSplitTrail.getInstance().layer, 'layerSplitAfter layers need to be different' );
          }
        }
      } );
      
      return true; // so we can assert( layerAudit() )
    },
    
    getDebugHTML: function() {
      var startPointer = new scenery.TrailPointer( new scenery.Trail( this ), true );
      var endPointer = new scenery.TrailPointer( new scenery.Trail( this ), false );
      
      function str( ob ) {
        return ob ? ob.toString() : ob;
      }
      
      var depth = 0;
      
      var result = '';
      
      var layerStartEntries = {};
      var layerEndEntries = {};
      _.each( this.layers, function( layer ) {
        var startIdx = layer.startPaintedTrail.getUniqueId();
        var endIndex = layer.endPaintedTrail.getUniqueId();
        layerStartEntries[startIdx] = '';
        layerEndEntries[endIndex] = '';
        layer.startPaintedTrail.reindex();
        layer.endPaintedTrail.reindex();
        var layerInfo = layer.getId() + ' <strong>' + layer.type.name + '</strong>' +
                        ' trails: ' + ( layer.startPaintedTrail ? str( layer.startPaintedTrail ) : layer.startPaintedTrail ) +
                        ',' + ( layer.endPaintedTrail ? str( layer.endPaintedTrail ) : layer.endPaintedTrail );
        layerInfo += '<span style="color: #008">';
        if ( layer.canUseDirtyRegions && !layer.canUseDirtyRegions() ) { layerInfo += ' dirtyRegionsDisabled'; }
        if ( layer.cssTranslation ) { layerInfo += ' cssTranslation'; }
        if ( layer.cssRotation ) { layerInfo += ' cssTranslation'; }
        if ( layer.cssScale ) { layerInfo += ' cssTranslation'; }
        if ( layer.cssTransform ) { layerInfo += ' cssTranslation'; }
        if ( layer.dirtyBounds && layer.dirtyBounds.isFinite() ) { layerInfo += ' dirtyBounds:' + layer.dirtyBounds.toString(); }
        layerInfo += '</span>';
        layerStartEntries[startIdx] += '<div style="color: #080">+Layer ' + layerInfo + '</div>';
        layerEndEntries[endIndex] += '<div style="color: #800">-Layer ' + layerInfo + '</div>';
      } );
      
      startPointer.depthFirstUntil( endPointer, function( pointer ) {
        var div;
        var ptr = str( pointer );
        var node = pointer.trail.lastNode();
        
        function addQualifier( text ) {
            div += ' <span style="color: #008">' + text + '</span>';
          }
        
        if ( pointer.isBefore && layerStartEntries[pointer.trail.getUniqueId()] ) {
          result += layerStartEntries[pointer.trail.getUniqueId()];
        }
        if ( pointer.isBefore ) {
          div = '<div style="margin-left: ' + ( depth * 20 ) + 'px">';
          if ( node.constructor.name ) {
            div += ' ' + node.constructor.name; // see http://stackoverflow.com/questions/332422/how-do-i-get-the-name-of-an-objects-type-in-javascript
          }
          div += ' <span style="font-weight: ' + ( node.isPainted() ? 'bold' : 'normal' ) + '">' + pointer.trail.lastNode().getId() + '</span>';
          div += ' <span style="color: #888">' + str( pointer.trail ) + '</span>';
          if ( !node._visible ) {
            addQualifier( 'invisible' );
          }
          if ( !node._pickable ) {
            addQualifier( 'unpickable' );
          }
          if ( node._clipShape ) {
            addQualifier( 'clipShape' );
          }
          if ( node._renderer ) {
            addQualifier( 'renderer:' + node._renderer.name );
          }
          if ( node._rendererOptions ) {
            // addQualifier( 'rendererOptions:' + _.each( node._rendererOptions, function( option, key ) { return key + ':' + str( option ); } ).join( ',' ) );
          }
          if ( node._layerSplitBefore ) {
            addQualifier( 'layerSplitBefore' );
          }
          if ( node._layerSplitAfter ) {
            addQualifier( 'layerSplitAfter' );
          }
          if ( node._opacity < 1 ) {
            addQualifier( 'opacity:' + node._opacity );
          }
          
          var transformType = '';
          switch ( node.transform.getMatrix().type ) {
            case Matrix3.Types.IDENTITY:       transformType = '';           break;
            case Matrix3.Types.TRANSLATION_2D: transformType = 'translated'; break;
            case Matrix3.Types.SCALING:        transformType = 'scale';      break;
            case Matrix3.Types.AFFINE:         transformType = 'affine';     break;
            case Matrix3.Types.OTHER:          transformType = 'other';      break;
          }
          if ( transformType ) {
            div += ' <span style="color: #88f" title="' + node.transform.getMatrix().toString().replace( '\n', '&#10;' ) + '">' + transformType + '</span>';
          }
          div += '</div>';
          result += div;
        }
        if ( pointer.isAfter && layerEndEntries[pointer.trail.getUniqueId()] ) {
          result += layerEndEntries[pointer.trail.getUniqueId()];
        }
        depth += pointer.isBefore ? 1 : -1;
      }, false );
      
      return result;
    },
    
    popupDebug: function() {
      var htmlContent = '<!DOCTYPE html>' +
                        '<html lang="en">' +
                        '<head><title>Scenery Debug Snapshot</title></head>' +
                        '<body>' + this.getDebugHTML() + '</body>' +
                        '</html>';
      window.open( 'data:text/html;charset=utf-8,' + encodeURIComponent( htmlContent ) );
    },
    
    getBasicConstructor: function( propLines ) {
      return 'new scenery.Scene( $( \'#main\' ), {' + propLines + '} )';
    },
    
    toStringWithChildren: function( mutateScene ) {
      var scene = this;
      var result = '';
      
      var nodes = this.getTopologicallySortedNodes().slice( 0 ).reverse(); // defensive slice, in case we store the order somewhere
      
      function name( node ) {
        return node === scene ? 'scene' : node.constructor.name.toLowerCase() + node.id;
      }
      
      _.each( nodes, function( node ) {
        if ( result ) {
          result += '\n';
        }
        
        if ( mutateScene && node === scene ) {
          var props = scene.getPropString( '  ', false );
          result += 'scene.mutate( {' + ( props ? ( '\n' + props + '\n' ) : '' ) + '} )';
        } else {
          result += 'var ' + name( node ) + ' = ' + node.toString( '', false );
        }
        
        _.each( node.children, function( child ) {
          result += '\n' + name( node ) + '.addChild( ' + name( child ) + ' );';
        } );
      } );
      
      return result;
    }
  } );
  
  function applyCSSHacks( $main, options ) {
    // to use CSS3 transforms for performance, hide anything outside our bounds by default
    if ( !options.allowSceneOverflow ) {
      $main.css( 'overflow', 'hidden' );
    }
    
    // forward all pointer events
    $main.css( '-ms-touch-action', 'none' );
    
    if ( options.allowCSSHacks ) {
      // some css hacks (inspired from https://github.com/EightMedia/hammer.js/blob/master/hammer.js)
      (function() {
        var prefixes = [ '-webkit-', '-moz-', '-ms-', '-o-', '' ];
        var properties = {
          userSelect: 'none',
          touchCallout: 'none',
          touchAction: 'none',
          userDrag: 'none',
          tapHighlightColor: 'rgba(0,0,0,0)'
        };
        
        _.each( prefixes, function( prefix ) {
          _.each( properties, function( propertyValue, propertyName ) {
            $main.css( prefix + propertyName, propertyValue );
          } );
        } );
      })();
    }
  }
  
  return Scene;
} );
