// Copyright 2002-2012, University of Colorado

/**
 * Main scene, that is also a Node.
 *
 * TODO: documentation!
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  var Vector2 = require( 'DOT/Vector2' );
  
  var scenery = require( 'SCENERY/scenery' );
  
  var Node = require( 'SCENERY/nodes/Node' ); // inherits from Node
  require( 'SCENERY/util/Trail' );
  require( 'SCENERY/util/TrailInterval' );
  require( 'SCENERY/util/TrailPointer' );
  require( 'SCENERY/input/Input' );
  require( 'SCENERY/layers/LayerBuilder' );
  require( 'SCENERY/layers/Renderer' );
  
  var Util = require( 'SCENERY/util/Util' );
  var objectCreate = Util.objectCreate;
  
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
    assert && assert( $main[0], 'A main container is required for a scene' );
    
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
    
    // main layers in a scene
    this.layers = [];
    
    this.layerChangeIntervals = []; // array of {TrailInterval}s indicating what parts need to be stitched together
    
    this.lastCursor = null;
    this.defaultCursor = $main.css( 'cursor' );
    
    this.$main = $main;
    // resize the main container as a sanity check
    this.setSize( options.width, options.height );
    
    this.sceneBounds = new Bounds2( 0, 0, $main.width(), $main.height() );
    
    // default to a canvas layer type, but this can be changed
    this.preferredSceneLayerType = options.preferredSceneLayerType;
    
    applyCSSHacks( $main, options );
    
    // note, arguments to the functions are mutable. don't destroy them
    this.sceneEventListener = {
      markForLayerRefresh: function( args ) { // contains trail
        console.log( 'marking layer refresh: ' + args.trail.toString() );
        scene.markInterval( args.trail );
      },
      
      markForInsertion: function( args ) { // contains parent, child, index, trail
        var affectedTrail = args.trail.copy().addDescendant( args.child );
        console.log( 'marking insertion: ' + affectedTrail.toString() );
        scene.markInterval( affectedTrail );
      },
      
      markForRemoval: function( args ) { // contains parent, child, index, trail
        // mark the interval
        var affectedTrail = args.trail.copy().addDescendant( args.child );
        console.log( 'marking removal: ' + affectedTrail.toString() );
        scene.markInterval( affectedTrail );
        
        // signal to the relevant layers to remove the specified trail while the trail is still valid.
        // waiting until after the removal takes place would require more complicated code to properly handle the trails
        affectedTrail.eachTrailUnder( function( trail ) {
          scene.layerLookup( trail ).removeNodeFromTrail( trail );
        } );
      },
      
      stitch: function( args ) { // contains match {Boolean}
        scene.stitch( args.match );
      },
      
      dirtyBounds: function( args ) { // contains node, bounds, transform, trail
        var trail = args.trail;
        
        // if there are no layers, no nodes would actually render, so don't do the lookup
        if ( scene.layers.length ) {
          _.each( scene.affectedLayers( trail ), function( layer ) {
            layer.markDirtyRegion( args );
          } );
        }
      },
      
      transform: function( args ) { // conatins node, type, matrix, transform, trail
        var trail = args.trail;
        
        if ( scene.layers.length ) {
          _.each( scene.affectedLayers( trail ), function( layer ) {
            layer.transformChange( args );
          } );
        }
      }
    };
    
    this.addEventListener( this.sceneEventListener );
  };
  var Scene = scenery.Scene;

  Scene.prototype = objectCreate( Node.prototype );
  Scene.prototype.constructor = Scene;
  
  Scene.prototype.updateScene = function( args ) {
    // validating bounds, similar to Piccolo2d
    this.validateBounds();
    this.validatePaint();
    
    // bail if there are no layers. consider a warning?
    if ( !this.layers.length ) {
      return;
    }
    
    var scene = this;
    
    _.each( this.layers, function( layer ) {
      layer.render( scene, args );
    } );
    
    this.updateCursor();
  };
  
  Scene.prototype.renderScene = function() {
    // TODO: for now, go with the same path. possibly add options later
    this.updateScene();
  };
  
  Scene.prototype.markInterval = function( affectedTrail ) {
    // since this is marked while the child is still connected, we can use our normal trail handling.
    
    // find the closest before and after self trails that are not affected
    var beforeTrail = affectedTrail.previousSelf(); // easy for the before trail
    
    var afterTrailPointer = new scenery.TrailPointer( affectedTrail.copy(), false );
    while ( afterTrailPointer.hasTrail() && ( !afterTrailPointer.isBefore || !afterTrailPointer.trail.lastNode().hasSelf() ) ) {
      afterTrailPointer.nestedForwards();
    }
    var afterTrail = afterTrailPointer.trail;
    
    // store the layer of the before/after trails so that it is easy to access later
    this.addLayerChangeInterval( new scenery.TrailInterval(
      beforeTrail,
      afterTrail,
      beforeTrail ? this.layerLookup( beforeTrail ) : null,
      afterTrail ? this.layerLookup( afterTrail ) : null
    ) );
  };
  
  // convenience function for layer change intervals
  Scene.prototype.addLayerChangeInterval = function( interval ) {
    console.log( 'adding interval: ' + interval.toString() );
    // TODO: replace with a binary-search-like version that may be faster. this includes a full scan
    
    // attempt to merge this interval with another if possible.
    for ( var i = 0; i < this.layerChangeIntervals.length; i++ ) {
      var other = this.layerChangeIntervals[i];
      other.reindex(); // sanity check, although for most use-cases this should be unnecessary
      
      if ( interval.exclusiveUnionable( other ) ) {
        // the interval can be unioned without including other nodes. do this, and remove the other interval from consideration
        interval = interval.union( other );
        this.layerChangeIntervals.splice( i, 1 );
      }
    }
    
    this.layerChangeIntervals.push( interval );
  };
  
  // insert a layer into the proper place (from its starting boundary)
  Scene.prototype.insertLayer = function( layer ) {
    for ( var i = 0; i < this.layers.length; i++ ) {
      // compare end and start boundaries, as they should match
      if ( this.layers[i].endBoundary.nextSelfTrail.equals( layer.startBoundary.nextSelfTrail ) ) {
        break;
      }
    }
    if ( i < this.layers.length ) {
      this.layers.splice( i + 1, 0, layer );
    } else {
      this.layers.push( layer );
    }
  };
  
  Scene.prototype.stitch = function( match ) {
    var scene = this;
    
    // we need to map old layer IDs to new layers if we 'glue' two layers into one, so that the layer references we put on the
    // intervals can be mapped to current layers.
    var layerMap = {};
    
    // default arguments for constructing layers
    var layerArgs = {
      $main: this.$main,
      scene: this,
      baseNode: this
    };
    
    console.log( 'stitching on intervals: \n' + this.layerChangeIntervals.join( '\n' ) );
    
    _.each( this.layerChangeIntervals, function( interval ) {
      console.log( 'before reindex: ' + interval.toString() );
      interval.reindex();
      console.log( 'stitch on interval ' + interval.toString() );
      var beforeTrail = interval.a;
      var afterTrail = interval.b;
      
      // stored here, from in markInterval
      var beforeLayer = interval.dataA;
      var afterLayer = interval.dataB;
      
      // if these layers are out of date, update them. 'while' will handle chained updates. circular references should be impossible
      while ( beforeLayer && layerMap[beforeLayer.getId()] ) {
        beforeLayer = layerMap[beforeLayer.getId()];
      }
      while ( afterLayer && layerMap[afterLayer.getId()] ) {
        afterLayer = layerMap[afterLayer.getId()];
      }
      
      var builder = new scenery.LayerBuilder( scene, beforeLayer ? beforeLayer.type : null, beforeTrail, afterTrail );
      
      // push the preferred layer type before we push that for any nodes
      if ( scene.preferredSceneLayerType ) {
        builder.pushPreferredLayerType( scene.preferredSceneLayerType );
      }
      
      builder.run();
      
      var boundaries = builder.boundaries;
      
      if ( match ) {
        // TODO: patch in the matching version!
        // scene.stitchInterval( layerMap, layerArgs, beforeTrail, afterTrail, beforeLayer, afterLayer, boundaries );
        this.rebuildLayers(); // bleh
      } else {
        scene.stitchInterval( layerMap, layerArgs, beforeTrail, afterTrail, beforeLayer, afterLayer, boundaries, match );
      }
      
      // console.log( '---' );
      // console.log( 'boundaries:' );
      // _.each( boundaries, function( boundary ) {
      //   console.log( 'boundary:' );
      //   console.log( '    types:    ' + ( boundary.hasPrevious() ? boundary.previousLayerType.name : '' ) + ' => ' + ( boundary.hasNext() ? boundary.nextLayerType.name : '' ) );
      //   console.log( '    trails:   ' + ( boundary.hasPrevious() ? boundary.previousSelfTrail.getUniqueId() : '' ) + ' => ' + ( boundary.hasNext() ? boundary.nextSelfTrail.getUniqueId() : '' ) );
      //   console.log( '    pointers: ' + ( boundary.hasPrevious() ? boundary.previousEndPointer.toString() : '' ) + ' => ' + ( boundary.hasNext() ? boundary.nextStartPointer.toString() : '' ) );
      // } );
    } );
    this.layerChangeIntervals = [];
    
    this.reindexLayers();
  }
  
  Scene.prototype.stitchInterval = function( layerMap, layerArgs, beforeTrail, afterTrail, beforeLayer, afterLayer, boundaries, match ) {
    var scene = this;
    
    var beforeLayerIndex = _.indexOf( this.layers, beforeLayer );
    var afterLayerIndex = _.indexOf( this.layers, afterLayer );
    
    var beforePointer = beforeTrail ? new scenery.TrailPointer( beforeTrail, true ) : new scenery.TrailPointer( new scenery.Trail( this ), true );
    var afterPointer = afterTrail ? new scenery.TrailPointer( afterTrail, true ) : new scenery.TrailPointer( new scenery.Trail( this ), false );
    
    /*---------------------------------------------------------------------------*
    * State
    *----------------------------------------------------------------------------*/
    
    var nextBoundaryIndex = 0;
    var nextBoundary = boundaries[nextBoundaryIndex];
    var trailsToAddToLayer = [];
    var currentTrail = beforeTrail;
    var currentLayer = beforeLayer;
    var currentLayerType = beforeLayer ? beforeLayer.type : null;
    var currentStartBoundary = null;
    
    // a list of layers that are most likely removed, not including the afterLayer for gluing
    var layersToRemove = [];
    for ( var i = beforeLayerIndex + 1; i < afterLayerIndex; i++ ) {
      layersToRemove.push( this.layers[i] );
    }
    
    function addPendingTrailsToLayer() {
      // add the necessary nodes to the layer
      _.each( trailsToAddToLayer, function( trail ) {
        currentLayer.addNodeFromTrail( trail );
      } );
      trailsToAddToLayer = [];
    }
    
    function step( trail ) {
      console.log( 'step: ' + ( trail ? trail.toString() : trail ) );
      // check for a boundary at this step between currentTrail and trail
      
      // if there is no next boundary, don't bother checking anyways
      if ( nextBoundary && ( ( nextBoundary.previousSelfTrail && currentTrail )
                             ? nextBoundary.previousSelfTrail.equals( currentTrail ) // non-null style check
                             : nextBoundary.previousSelfTrail === currentTrail ) ) { // at least one null check
        assert && assert( ( nextBoundary.nextSelfTrail && trail )
                          ? nextBoundary.nextSelfTrail.equals( trail )
                          : nextBoundary.nextSelfTrail === trail );
        // console.log( 'step boundary:' );
        // console.log( '    types:    ' + ( nextBoundary.hasPrevious() ? nextBoundary.previousLayerType.name : '' ) + ' => ' + ( nextBoundary.hasNext() ? nextBoundary.nextLayerType.name : '' ) );
        // console.log( '    trails:   ' + ( nextBoundary.hasPrevious() ? nextBoundary.previousSelfTrail.getUniqueId() : '' ) + ' => ' + ( nextBoundary.hasNext() ? nextBoundary.nextSelfTrail.getUniqueId() : '' ) );
        // console.log( '    pointers: ' + ( nextBoundary.hasPrevious() ? nextBoundary.previousEndPointer.toString() : '' ) + ' => ' + ( nextBoundary.hasNext() ? nextBoundary.nextStartPointer.toString() : '' ) );
        
        // we are at a boundary change. verify that we are at the end of a layer
        if ( currentLayer || currentStartBoundary ) {
          if ( currentLayer ) {
            console.log( 'has currentLayer' );
            // existing layer, reposition its endpoint
            currentLayer.endBoundary = nextBoundary;
            // TODO: fix up layer so these extra sets are not necessary?
            currentLayer.endPointer = nextBoundary.previousEndPointer;
            currentLayer.endSelfTrail = nextBoundary.previousSelfTrail;
          } else {
            console.log( 'creating layer' );
            assert && assert( currentStartBoundary );
            currentLayer = currentLayerType.createLayer( _.extend( {
              startBoundary: currentStartBoundary,
              endBoundary: nextBoundary
            }, layerArgs ) );
            currentLayer.type = currentLayerType;
            scene.insertLayer( currentLayer );
          }
          // sanity checks
          assert && assert( currentLayer.startSelfTrail );
          assert && assert( currentLayer.endSelfTrail );
          
          addPendingTrailsToLayer();
        } else {
          // if not at the end of a layer, sanity check that we should have no accumulated pending trails
          console.log( 'was first layer' );
          assert && assert( trailsToAddToLayer.length === 0 );
        }
        currentLayer = null;
        currentLayerType = nextBoundary.nextLayerType;
        currentStartBoundary = nextBoundary;
        nextBoundaryIndex++;
        nextBoundary = boundaries[nextBoundaryIndex];
      }
      if ( trail ) {
        trailsToAddToLayer.push( trail );
      }
      currentTrail = trail;
    }
    
    function startStep( trail ) {
      console.log( 'startStep: ' + ( trail ? trail.toString() : trail ) );
    }
    
    function middleStep( trail ) {
      console.log( 'middleStep: ' + trail.toString() );
      step( trail );
    }
    
    function endStep( trail ) {
      console.log( 'endStep: ' + ( trail ? trail.toString() : trail ) );
      step( trail );
      
      // TODO: better handling and concepts of beforeLayer / afterLayer when endtrails are null. leaving superfluous layer after removing everything
      if ( beforeLayer !== afterLayer && boundaries.length === 0 ) {
        // glue the layers together
        beforeLayer.endBoundary = afterLayer.endBoundary;
        layersToRemove.push( afterLayer );
        currentLayer = beforeLayer;
        addPendingTrailsToLayer();
        
        // move over all of afterLayer's trails to beforeLayer
        // defensive copy needed, since this will be modified at the same time
        _.each( afterLayer._layerTrails.slice( 0 ), function( trail ) {
          afterLayer.removeNodeFromTrail( trail );
          beforeLayer.addNodeFromTrail( trail );
        } );
        
        layerMap[afterLayer.getId()] = beforeLayer;
        
        scene.disposeLayer( afterLayer );
      } else {
        currentLayer = afterLayer;
        if ( currentLayer ) {
          currentLayer.startBoundary = currentStartBoundary;
          // TODO: fix up layer so these extra sets are not necessary?
          currentLayer.startPointer = currentStartBoundary ? currentStartBoundary.nextEndPointer : null;
          currentLayer.startSelfTrail = currentStartBoundary ? currentStartBoundary.nextSelfTrail : null;
        }
        
        addPendingTrailsToLayer();
      }
      
      _.each( layersToRemove, function( layer ) {
        scene.disposeLayer( layer );
      } );
    }
    
    // iterate from beforeTrail up to BEFORE the afterTrail. does not include afterTrail
    startStep( beforeTrail );
    beforePointer.eachTrailBetween( afterPointer, function( trail ) {
      // ignore non-self trails
      if ( !trail.lastNode().hasSelf() || ( beforeTrail && trail.equals( beforeTrail ) ) ) {
        return;
      }
      
      middleStep( trail.copy() );
    } );
    endStep( afterTrail );
  };
  
  Scene.prototype.rebuildLayers = function() {
    console.log( 'rebuildLayers' );
    // remove all of our tracked layers from the container, so we can fill it with fresh layers
    this.disposeLayers();
    
    var builder = new scenery.LayerBuilder( this, null, null, null );
    
    if ( this.preferredSceneLayerType ) {
      builder.pushPreferredLayerType( this.preferredSceneLayerType );
    }
    
    builder.run();
    
    this.boundaries = builder.boundaries;
    
    var layerArgs = {
      $main: this.$main,
      scene: this,
      baseNode: this
    };
    
    this.layers = [];
    
    // console.log( this.boundaries );
    
    for ( var i = 1; i < this.boundaries.length; i++ ) {
      var startBoundary = this.boundaries[i-1];
      var endBoundary = this.boundaries[i];
      
      assert && assert( startBoundary.nextLayerType === endBoundary.previousLayerType );
      var layerType = startBoundary.nextLayerType;
      
      // LayerType is responsible for applying its own arguments in createLayer()
      var layer = layerType.createLayer( _.extend( {
        startBoundary: startBoundary,
        endBoundary: endBoundary
      }, layerArgs ) );
      
      // record the type on the layer
      layer.type = layerType;
      
      // add the initial nodes to the layer
      layer.startPointer.eachTrailBetween( layer.endPointer, function( trail ) {
        if ( trail.lastNode().hasSelf() ) {
          layer.addNodeFromTrail( trail );
        }
      } );
      
      this.layers.push( layer );
    }
    
    // console.log( '---' );
    // console.log( 'boundaries:' );
    // _.each( this.boundaries, function( boundary ) {
    //   console.log( 'boundary:' );
    //   console.log( '    types:    ' + ( boundary.hasPrevious() ? boundary.previousLayerType.name : '' ) + ' => ' + ( boundary.hasNext() ? boundary.nextLayerType.name : '' ) );
    //   console.log( '    trails:   ' + ( boundary.hasPrevious() ? boundary.previousSelfTrail.getUniqueId() : '' ) + ' => ' + ( boundary.hasNext() ? boundary.nextSelfTrail.getUniqueId() : '' ) );
    //   console.log( '    pointers: ' + ( boundary.hasPrevious() ? boundary.previousEndPointer.toString() : '' ) + ' => ' + ( boundary.hasNext() ? boundary.nextStartPointer.toString() : '' ) );
    // } );
    // console.log( 'layers:' );
    // _.each( this.layers, function( layer ) {
    //   console.log( layer.toString() );
    // } );
  };
  
  // after layer changes, the layers should have their zIndex updated
  Scene.prototype.reindexLayers = function() {
    var index = 1;
    _.each( this.layers, function( layer ) {
      // layers increment indices as needed
      index = layer.reindex( index );
    } );
  };
  
  Scene.prototype.dispose = function() {
    this.disposeLayers();
    
    // TODO: clear event handlers if added
    //throw new Error( 'unimplemented dispose: clear event handlers if added' );
  };
  
  Scene.prototype.disposeLayer = function( layer ) {
    layer.dispose();
    this.layers.splice( _.indexOf( this.layers, layer ), 1 ); // TODO: better removal code!
  };
  
  Scene.prototype.disposeLayers = function() {
    var scene = this;
    
    _.each( this.layers.slice( 0 ), function( layer ) {
      scene.disposeLayer( layer );
    } );
  };
  
  // what layer does this trail's terminal node render in? returns null if the node is not contained in a layer
  Scene.prototype.layerLookup = function( trail ) {
    // TODO: add tree form for optimization! this is slower than necessary, it shouldn't be O(n)!
    assert && assert( !( trail.isEmpty() || trail.nodes[0] !== this ), 'layerLookup root matches' );
    assert && assert( trail.lastNode().hasSelf(), 'layerLookup only supports nodes with hasSelf(), as this guarantees an unambiguous answer' );
    
    if ( this.layers.length === 0 ) {
      return null; // node not contained in a layer
    }
    
    for ( var i = 0; i < this.layers.length; i++ ) {
      var layer = this.layers[i];
      
      // trails may be stale, so we need to update their indices
      if ( layer.startSelfTrail ) { layer.startSelfTrail.reindex(); }
      if ( layer.endSelfTrail ) { layer.endSelfTrail.reindex(); }
      
      if ( !layer.endSelfTrail || trail.compare( layer.endSelfTrail ) <= 0 ) {
        if ( !layer.startSelfTrail || trail.compare( layer.startSelfTrail ) >= 0 ) {
          return layer;
        } else {
          return null; // node is not contained in a layer (it is before any existing layer)
        }
      }
    }
    
    return null; // node not contained in a layer (it is after any existing layer)
  };
  
  // all layers whose start or end points lie inclusively in the range from the trail's before and after
  Scene.prototype.affectedLayers = function( trail ) {
    // TODO: add tree form for optimization! this is slower than necessary, it shouldn't be O(n)!
    
    var result = [];
    
    assert && assert( !( trail.isEmpty() || trail.nodes[0] !== this ), 'layerLookup root matches' );
    
    if ( this.layers.length === 0 ) {
      throw new Error( 'no layers in the scene' );
    }
    
    // point to the beginning of the node, right before it would be rendered
    var startPointer = new scenery.TrailPointer( trail, true );
    var endPointer = new scenery.TrailPointer( trail, false );
    
    for ( var i = 0; i < this.layers.length; i++ ) {
      var layer = this.layers[i];
      
      var notBefore = endPointer.compareNested( new scenery.TrailPointer( layer.startSelfTrail, true ) ) !== -1;
      var notAfter = startPointer.compareNested( new scenery.TrailPointer( layer.endSelfTrail, true ) ) !== 1;
      
      if ( notBefore && notAfter ) {
        result.push( layer );
      }
    }
    
    return result;
  };
  
  // attempt to render everything currently visible in the scene to an external canvas. allows copying from canvas layers straight to the other canvas
  Scene.prototype.renderToCanvas = function( canvas, context, callback ) {
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
  };
  
  // TODO: consider SVG data URLs
  
  Scene.prototype.canvasDataURL = function( callback ) {
    this.canvasSnapshot( function( canvas ) {
      callback( canvas.toDataURL() );
    } );
  };
  
  // renders what it can into a Canvas (so far, Canvas and SVG layers work fine)
  Scene.prototype.canvasSnapshot = function( callback ) {
    var canvas = document.createElement( 'canvas' );
    canvas.width = this.sceneBounds.getWidth();
    canvas.height = this.sceneBounds.getHeight();
    
    var context = canvas.getContext( '2d' );
    this.renderToCanvas( canvas, context, function() {
      callback( canvas, context.getImageData( 0, 0, canvas.width, canvas.height ) );
    } );
  };
  
  Scene.prototype.setSize = function( width, height ) {
    // resize our main container
    this.$main.width( width );
    this.$main.height( height );
    
    // set the container's clipping so anything outside won't show up
    // TODO: verify this clipping doesn't reduce performance!
    this.$main.css( 'clip', 'rect(0px,' + width + 'px,' + height + 'px,0px)' );
    
    this.sceneBounds = new Bounds2( 0, 0, width, height );
  };
  
  Scene.prototype.resize = function( width, height ) {
    this.setSize( width, height );
    this.rebuildLayers(); // TODO: why?
  };
  
  Scene.prototype.getSceneWidth = function() {
    return this.sceneBounds.getWidth();
  };
  
  Scene.prototype.getSceneHeight = function() {
    return this.sceneBounds.getHeight();
  };
  
  Scene.prototype.updateCursor = function() {
    if ( this.input && this.input.mouse.point ) {
      var mouseTrail = this.trailUnderPoint( this.input.mouse.point );
      
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
  };
  
  Scene.prototype.setSceneCursor = function( cursor ) {
    if ( cursor !== this.lastCursor ) {
      this.lastCursor = cursor;
      this.$main.css( 'cursor', cursor );
    }
  };
  
  Scene.prototype.initializeStandaloneEvents = function( parameters ) {
    // TODO extract similarity between standalone and fullscreen!
    var element = this.$main[0];
    this.initializeEvents( _.extend( {}, {
      listenerTarget: element,
      pointFromEvent: function( evt ) {
        var mainBounds = element.getBoundingClientRect();
        return new Vector2( evt.clientX - mainBounds.left, evt.clientY - mainBounds.top );
      }
    }, parameters ) );
  };
  
  Scene.prototype.initializeFullscreenEvents = function( parameters ) {
    var element = this.$main[0];
    this.initializeEvents( _.extend( {}, {
      listenerTarget: document,
      pointFromEvent: function( evt ) {
        var mainBounds = element.getBoundingClientRect();
        return new Vector2( evt.clientX - mainBounds.left, evt.clientY - mainBounds.top );
      }
    }, parameters ) );
  };
  
  Scene.prototype.initializeEvents = function( parameters ) {
    var scene = this;
    
    if ( scene.input ) {
      throw new Error( 'Attempt to attach events twice to the scene' );
    }
    
    // TODO: come up with more parameter names that have the same string length, so it looks creepier
    var pointFromEvent = parameters.pointFromEvent;
    var listenerTarget = parameters.listenerTarget;
    
    var input = new scenery.Input( scene, listenerTarget );
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
  };
  
  Scene.prototype.resizeOnWindowResize = function() {
    var scene = this;
    
    var resizer = function () {
      scene.resize( window.innerWidth, window.innerHeight );
    };
    $( window ).resize( resizer );
    resizer();
  };
  
  Scene.prototype.getDebugHTML = function() {
    var startPointer = new scenery.TrailPointer( new scenery.Trail( this ), true );
    var endPointer = new scenery.TrailPointer( new scenery.Trail( this ), false );
    
    var depth = 0;
    
    var result = '';
    
    var layerEntries = [];
    _.each( this.layers, function( layer ) {
      layer.startPointer.trail.reindex();
      layer.endPointer.trail.reindex();
      var startIdx = layer.startPointer.toString();
      var endIndex = layer.endPointer.toString();
      if ( !layerEntries[startIdx] ) {
        layerEntries[startIdx] = '';
      }
      if ( !layerEntries[endIndex] ) {
        layerEntries[endIndex] = '';
      }
      layer.startSelfTrail.reindex();
      layer.endSelfTrail.reindex();
      var layerInfo = layer.getId() + ' <strong>' + layer.type.name + '</strong>' +
                      ' trails: ' + ( layer.startSelfTrail ? layer.startSelfTrail.toString() : layer.startSelfTrail ) +
                      ',' + ( layer.endSelfTrail ? layer.endSelfTrail.toString() : layer.endSelfTrail ) +
                      ' pointers: ' + layer.startPointer.toString() +
                      ',' + layer.endPointer.toString();
      layerEntries[startIdx] += '<div style="color: #080">+Layer ' + layerInfo + '</div>';
      layerEntries[endIndex] += '<div style="color: #800">-Layer ' + layerInfo + '</div>';
    } );
    
    startPointer.depthFirstUntil( endPointer, function( pointer ) {
      var div;
      var ptr = pointer.toString();
      var node = pointer.trail.lastNode();
      
      function addQualifier( text ) {
          div += ' <span style="color: #008">' + text + '</span>';
        }
      
      if ( layerEntries[ptr] ) {
        result += layerEntries[ptr];
      }
      if ( pointer.isBefore ) {
        div = '<div style="margin-left: ' + ( depth * 20 ) + 'px">';
        if ( node.constructor.name ) {
          div += ' ' + node.constructor.name;
        }
        div += ' <span style="font-weight: ' + ( node.hasSelf() ? 'bold' : 'normal' ) + '">' + pointer.trail.lastNode().getId() + '</span>';
        div += ' <span style="color: #888">' + pointer.trail.toString() + '</span>';
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
          addQualifier( 'rendererOptions:' + _.each( node._rendererOptions, function( option, key ) { return key + ':' + option ? option.toString() : option; } ).join( ',' ) );
        }
        if ( node._layerSplitBefore ) {
          addQualifier( 'layerSplitBefore' );
        }
        if ( node._layerSplitAfter ) {
          addQualifier( 'layerSplitAfter' );
        }
        div += '</div>';
        result += div;
      }
      depth += pointer.isBefore ? 1 : -1;
    }, false );
    
    return result;
  };
  
  Scene.prototype.popupDebug = function() {
    var htmlContent = '<!DOCTYPE html>'+
                      '<html lang="en">'+
                      '<head><title>Scenery Debug Snapshot</title></head>'+
                      '<body>' + this.getDebugHTML() + '</body>'+
                      '</html>';
    window.open( 'data:text/html;charset=utf-8,' + encodeURIComponent( htmlContent ) );
  };
  
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
