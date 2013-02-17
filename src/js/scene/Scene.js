// Copyright 2002-2012, University of Colorado

/**
 * Main scene
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.Scene = function( main, params ) {
    scenery.Node.call( this, params );
    
    var scene = this;
    
    // main layers in a scene
    this.layers = [];
    
    this.main = main;
    
    this.sceneBounds = new phet.math.Bounds2( 0, 0, main.width(), main.height() );
    
    // default to a canvas layer type, but this can be changed
    this.preferredSceneLayerType = scenery.LayerType.Canvas;
    
    applyCSSHacks( main );
    
    // note, arguments to the functions are mutable. don't destroy them
    this.sceneEventListener = {
      insertChild: function( args ) {
        var parent = args.parent;
        var child = args.child;
        var index = args.index;
        var trail = args.trail;
        
        // TODO: improve later
        scene.rebuildLayers();
      },
      
      removeChild: function( args ) {
        var parent = args.parent;
        var child = args.child;
        var index = args.index;
        var trail = args.trail;
        
        scene.rebuildLayers();
      },
      
      dirtyBounds: function( args ) {
        var node = args.node;
        var localBounds = args.bounds;
        var transform = args.transform;
        var trail = args.trail;
        
        scene.layerLookup( trail ).markDirtyRegion( node, localBounds, transform, trail );
      },
      
      layerRefresh: function( args ) {
        var node = args.node;
        var trail = args.trail;
        
        scene.rebuildLayers();
      }
    };
    
    this.addEventListener( this.sceneEventListener );
  };

  var Scene = scenery.Scene;
  
  Scene.prototype = phet.Object.create( scenery.Node.prototype );
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
  };
  
  Scene.prototype.renderScene = function() {
    // TODO: for now, go with the same path. possibly add options later
    this.updateScene();
  };
  
  Scene.prototype.rebuildLayers = function() {
    // remove all of our tracked layers from the container, so we can fill it with fresh layers
    this.disposeLayers();
    
    // TODO: internal API rethink
    var state = new scenery.LayerState();
    
    if ( this.preferredSceneLayerType ) {
      state.pushPreferredLayerType( this.preferredSceneLayerType );
    }
    
    var layerEntries = state.buildLayers( new scenery.TrailPointer( new scenery.Trail( this ), true ), new scenery.TrailPointer( new scenery.Trail( this ), false ), null );
    
    var layerArgs = {
      main: this.main,
      scene: this
    };
    
    this.layers = _.map( layerEntries, function( entry ) {
      var layer = entry.type.createLayer( layerArgs );
      layer.updateBoundaries( entry );
      return layer;
    } );
    
    console.log( '---' );
    console.log( 'layers rebuilt:' );
    _.each( this.layers, function( layer ) {
      console.log( layer.toString() );
    } );
    _.each( layerEntries, function( entry ) {
      //console.log( entry.type.name + ' ' + ( this.startPointer ? this.startPointer.toString() : '!' ) + ' (' + ( this.startPath ? this.startPath.toString() : '!' ) + ') => ' + ( this.endPointer ? this.endPointer.toString() : '!' ) + ' (' + ( this.endPath ? this.endPath.toString() : '!' ) + ')' );
    } );
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
  
  Scene.prototype.disposeLayers = function() {
    var scene = this;
    
    _.each( this.layers.slice( 0 ), function( layer ) {
      layer.dispose();
      scene.layers.splice( _.indexOf( scene.layers, layer ), 1 ); // TODO: better removal code!
    } );
  };
  
  Scene.prototype.layerLookup = function( trail ) {
    // TODO: add tree form for optimization! this is slower than necessary, it shouldn't be O(n)!
    
    phet.assert( !( trail.isEmpty() || trail.nodes[0] !== this ), 'layerLookup root matches' );
    
    if ( this.layers.length === 0 ) {
      throw new Error( 'no layers in the scene' );
    }
    
    // point to the beginning of the node, right before it would be rendered
    var pointer = new scenery.TrailPointer( trail, true );
    
    for ( var i = 0; i < this.layers.length; i++ ) {
      var layer = this.layers[i];
      
      // the first layer whose end point is equal to or past our pointer should contain the trail
      if ( pointer.compareNested( layer.endPointer ) !== 1 ) {
        // TODO: consider removal for performance
        phet.assert( pointer.compareNested( layer.startPointer ) !== -1, 'node not contained in a layer' );
        return layer;
      }
    }
    
    throw new Error( 'node not contained in a layer' );
  };
  
  // attempt to render everything currently visible in the scene to an external canvas. allows copying from canvas layers straight to the other canvas
  // delayCounts will have increment() and decrement() called on it if asynchronous completion is needed.
  Scene.prototype.renderToCanvas = function( canvas, context, delayCounts ) {
    context.clearRect( 0, 0, canvas.width, canvas.height );
    _.each( this.layers, function( layer ) {
      layer.renderToCanvas( canvas, context, delayCounts );
    } );
  };
  
  Scene.prototype.resize = function( width, height ) {
    this.main.width( width );
    this.main.height( height );
    this.sceneBounds = new phet.math.Bounds2( 0, 0, width, height );
    this.rebuildLayers(); // TODO: why?
  };
  
  Scene.prototype.initializeStandaloneEvents = function() {
    var element = this.main[0];
    this.initializeEvents( {
      preventDefault: true,
      listenerTarget: element,
      pointFromEvent: function( evt ) {
        var mainBounds = element.getBoundingClientRect();
        return new phet.math.Vector2( evt.clientX - mainBounds.left, evt.clientY - mainBounds.top );
      }
    } );
  };
  
  Scene.prototype.initializeFullscreenEvents = function() {
    this.initializeEvents( {
      preventDefault: true,
      listenerTarget: document,
      pointFromEvent: function( evt ) {
        return new phet.math.Vector2( evt.pageX, evt.pageY );
      }
    } );
  };
  
  Scene.prototype.initializeEvents = function( parameters ) {
    var scene = this;
    
    // TODO: come up with more parameter names that have the same string length, so it looks creepier
    var pointFromEvent = parameters.pointFromEvent;
    var listenerTarget = parameters.listenerTarget;
    var preventDefault = parameters.preventDefault;
    
    var input = new scenery.Input( scene );
    
    $( listenerTarget ).on( 'mousedown', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseDown( pointFromEvent( evt ), evt );
    } );
    $( listenerTarget ).on( 'mouseup', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseUp( pointFromEvent( evt ), evt );
    } );
    $( listenerTarget ).on( 'mousemove', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseMove( pointFromEvent( evt ), evt );
    } );
    $( listenerTarget ).on( 'mouseover', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseOver( pointFromEvent( evt ), evt );
    } );
    $( listenerTarget ).on( 'mouseout', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      input.mouseOut( pointFromEvent( evt ), evt );
    } );

    function forEachChangedTouch( evt, callback ) {
      for ( var i = 0; i < evt.changedTouches.length; i++ ) {
        // according to spec (http://www.w3.org/TR/touch-events/), this is not an Array, but a TouchList
        var touch = evt.changedTouches.item( i );
        
        callback( touch.identifier, pointFromEvent( touch ) );
      }
    }

    $( listenerTarget ).on( 'touchstart', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      forEachChangedTouch( evt, function( id, point ) {
        input.touchStart( id, point, evt );
      } );
    } );
    $( listenerTarget ).on( 'touchend', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      forEachChangedTouch( evt, function( id, point ) {
        input.touchEnd( id, point, evt );
      } );
    } );
    $( listenerTarget ).on( 'touchmove', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      forEachChangedTouch( evt, function( id, point ) {
        input.touchMove( id, point, evt );
      } );
    } );
    $( listenerTarget ).on( 'touchcancel', function( jEvent ) {
      var evt = jEvent.originalEvent;
      if ( preventDefault ) { jEvent.preventDefault(); }
      forEachChangedTouch( evt, function( id, point ) {
        input.touchCancel( id, point, evt );
      } );
    } );
  };
  
  Scene.prototype.resizeOnWindowResize = function() {
    var scene = this;
    
    var resizer = function () {
      scene.resize( window.innerWidth, window.innerHeight );
    };
    $( window ).resize( resizer );
    resizer();
  };
    
  function applyCSSHacks( main ) {
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
          main.css( prefix + propertyName, propertyValue );
        } );
      } );
    })();
  }
})();
