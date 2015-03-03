// Copyright 2002-2014, University of Colorado Boulder

/*
 * Runs unit tests that compare past images of Scenery display output with the current display output.
 */
(function() {
  'use strict';

  module( 'Scenery: Pixel Comparison' );

  // We can only guarantee comparisons for Firefox and Chrome
  if ( !core.platform.firefox && !core.platform.chromium ) {
    window.console && window.console.log && window.console.log( 'Not running pixel-comparison tests' );
    return;
  }

  /**
   * Runs a pixel comparison test between a reference data URL and a Display (with options and setup).
   *
   * @param {string} name - Test name
   * @param {function} setup - Called to set up the scene and display with rendered content.
   *                           function( scene, display, asyncCallback ). If asynchronous, call the asyncCallback when
   *                           the Display is ready to be rasterized.
   * @param {string} dataURL - The reference data URL to compare against
   * @param {number} threshold - Numerical threshold to determine how much error is acceptable
   */
  function pixelTest( name, setup, dataURL, threshold, isAsync ) {
    asyncTest( name, function() {
      // set up the scene/display
      var scene = new scenery.Node();
      var display = new scenery.Display( scene, {
        preserveDrawingBuffer: true
      } );

      function loadImages() {
        // called when both images have been loaded
        function compareSnapshots() {
          var referenceSnapshot = snapshotFromImage( referenceImage );
          var freshSnapshot = snapshotFromImage( freshImage );

          // the actual comparison statement
          snapshotEquals( freshSnapshot, referenceSnapshot, threshold, name );

          // tell qunit that we're done? (that's what the old version did, seems potentially wrong but working?)
          start();
        }

        // load images to compare
        var loadedCount = 0;
        var referenceImage = document.createElement( 'img' );
        var freshImage = document.createElement( 'img' );
        referenceImage.onload = freshImage.onload = function() {
          if ( ++loadedCount === 2 ) {
            compareSnapshots();
          }
        };
        referenceImage.onerror = function() {
          ok( false, name + ' reference image failed to load' );
          start();
        };
        freshImage.onerror = function() {
          ok( false, name + ' fresh image failed to load' );
          start();
        };

        referenceImage.src = dataURL;

        display.foreignObjectRasterization( function( url ) {
          if ( !url ) {
            ok( false, name + ' failed to rasterize the display' );
            start();
            return;
          }
          freshImage.src = url;
        } );
      }

      setup( scene, display, loadImages );

      if ( !isAsync ) {
        loadImages();
      }
    } );
  }

  // Like pixelTest, but for multiple listeners ({string[]}). Don't override the renderer on the scene.
  function multipleRendererTest( name, setup, dataURL, threshold, renderers, isAsync ) {
    for ( var i = 0; i < renderers.length; i++ ) {
      (function(){
        var renderer = renderers[i];

        pixelTest( name + ' (' + renderer + ')', function( scene, display, asyncCallback ) {
          scene.renderer = renderer;
          setup( scene, display, asyncCallback );
        }, dataURL, threshold, isAsync );
      })();
    }
  }

  var simpleRectangleDataURL = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAAcElEQVRYR+3YwQoAIQhFUfv/j572NQRiQTOc1ipyn0+kFpe/dnl/ocGqQgieJPhUiyfzX9VcSazBgTCCyZGbwhFEcCRgzVgzVVcgiGDE8uS3ZpiESZgkNwMO1hyvORpBBD938lcl25Lv+62KEcHfE+wTtBwp2K8YwAAAAABJRU5ErkJggg==';
  multipleRendererTest( 'Simple Rectangle',
    function( scene, display ) {
      display.width = 40;
      display.height = 40;
      scene.addChild( new scenery.Rectangle( 6, 6, 28, 28, {
        fill: '#000000'
      } ) );
      display.updateDisplay();
    }, simpleRectangleDataURL,
    0, [ 'canvas', 'svg', 'dom', 'webgl', 'pixi' ]
  );

  multipleRendererTest( 'Shifted Rectangle',
    function( scene, display ) {
      display.width = 40;
      display.height = 40;
      scene.addChild( new scenery.Rectangle( 0, 0, 28, 28, {
        fill: '#000000',
        x: 6,
        y: 6
      } ) );
      display.updateDisplay();
    }, simpleRectangleDataURL,
    0, [ 'canvas', 'svg', 'dom', 'webgl', 'pixi' ]
  );

  multipleRendererTest( 'Delay-shifted Rectangle',
    function( scene, display ) {
      display.width = 40;
      display.height = 40;
      var rect = new scenery.Rectangle( 0, 0, 28, 28, {
        fill: '#000000',
        x: 10,
        y: -10
      } );
      scene.addChild( rect );
      display.updateDisplay();
      rect.x = 6;
      rect.y = 6;
      display.updateDisplay();
    }, simpleRectangleDataURL,
    0, [ 'canvas', 'svg', 'dom', 'webgl', 'pixi' ]
  );

  multipleRendererTest( 'Color-change Rectangle',
    function( scene, display ) {
      display.width = 40;
      display.height = 40;
      var rect = new scenery.Rectangle( 6, 6, 28, 28, {
        fill: 'green'
      } );
      scene.addChild( rect );
      display.updateDisplay();
      rect.fill = 'black';
      display.updateDisplay();
    }, simpleRectangleDataURL,
    0, [ 'canvas', 'svg', 'dom', 'webgl', 'pixi' ]
  );

  // try rendering the image itself
  multipleRendererTest( 'Image with PNG data URL',
    function( scene, display, asyncCallback ) {
      var img = document.createElement( 'img' );
      img.onload = function() {
        display.width = 40;
        display.height = 40;
        scene.addChild( new scenery.Image( img ) );
        display.updateDisplay();

        asyncCallback();
      };
      img.error = function() {
        asyncCallback();
      }
      img.src = simpleRectangleDataURL;
    }, simpleRectangleDataURL,
    0, [ 'canvas', 'svg', 'dom', 'webgl', 'pixi' ], true // asynchronous
  );

  multipleRendererTest( 'Invisible node with rectangles (paths) above and below',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var shape = kite.Shape.rectangle( 0, 0, 30, 30 );
      scene.addChild( new scenery.Path( shape, {
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: -10, y: -10
      } ) );
      scene.addChild( new scenery.Path( shape, {
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: 10, y: 10,
        visible: false
      } ) );
      scene.addChild( new scenery.Path( shape, {
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: 20, y: 20
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAWklEQVRYR+3WsQoAIAhF0ef/f3TZ0NAQWQgR3ajN5HkmTVLxl3J6I5P8xk4rJAACCCCAAAIIIPC5gI+fJhBbwsYqux7gJPXsTx8mvJF6o53aZVYCIIAAAk8KVE6CQBKyrnp4AAAAAElFTkSuQmCC',
    0, [ 'canvas', 'svg', 'dom', 'webgl', 'pixi' ]
  );

  multipleRendererTest( 'Invisible => Visible',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: '#f00', visible: false } );
      scene.addChild( rect );
      scene.addChild( new scenery.Rectangle( 0, 0, 32, 32, { fill: 'green', visible: false } ) );
      display.updateDisplay();
      rect.visible = true;
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3WMQoAMAgDwPj/R7e+wQpdTrIK4Ratk3TmU0lnPqUAAQIECBAg8F1gfsl3Np+eiY0KChAgQIAAAQIX6VUgIfXDabwAAAAASUVORK5CYII=',
    0, [ 'canvas', 'svg', 'dom', 'webgl', 'pixi' ]
  );

  multipleRendererTest( 'Invisible repaints',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: '#f00' } );
      scene.addChild( rect );
      display.updateDisplay();
      rect.visible = false;
      display.updateDisplay();
      rect.x = 16;
      rect.visible = true;
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3WMQoAMAgDQP3/o9tCf2AGlxPXQLhFu8I5VW/n0/PoTypAgAABAgQIrAuk5zz+BxQgQIAAAQIEUoELbg4gAWKut4YAAAAASUVORK5CYII=',
    0, [ 'canvas', 'svg', 'dom', 'webgl', 'pixi' ]
  );

  multipleRendererTest( 'Invisible Rectangle',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var rect = new scenery.Rectangle( 0, 0, 16, 16, { fill: '#f00', visible: false } );
      scene.addChild( rect );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAALUlEQVRYR+3QQREAAAABQfqXFsNnFTizzXk99+MAAQIECBAgQIAAAQIECBAgMBo/ACHo7lH9AAAAAElFTkSuQmCC',
    0, [ 'canvas', 'svg', 'dom', 'webgl', 'pixi' ]
  );
})();
