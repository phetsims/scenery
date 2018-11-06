// Copyright 2017, University of Colorado Boulder

/**
 * PixelComparison tests
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var Circle = require( 'SCENERY/nodes/Circle' );
  var Display = require( 'SCENERY/display/Display' );
  var Image = require( 'SCENERY/nodes/Image' );
  var LinearGradient = require( 'SCENERY/util/LinearGradient' );
  var Matrix3 = require( 'DOT/Matrix3' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Path = require( 'SCENERY/nodes/Path' );
  var Pattern = require( 'SCENERY/util/Pattern' );
  var platform = require( 'PHET_CORE/platform' );
  var Property = require( 'AXON/Property' );
  var RadialGradient = require( 'SCENERY/util/RadialGradient' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Shape = require( 'KITE/Shape' );
  var snapshotEquals = require( 'SCENERY/tests/snapshotEquals' );
  var Vector2 = require( 'DOT/Vector2' );

  QUnit.module( 'PixelComparison' );

  var DEFAULT_THRESHOLD = 1.5;

  function snapshotFromImage( image ) { // eslint-disable-line no-unused-vars

    var canvas = document.createElement( 'canvas' );
    canvas.width = image.width;
    canvas.height = image.height;
    var context = canvas.getContext( '2d' );
    context.drawImage( image, 0, 0, image.width, image.height );
    return context.getImageData( 0, 0, image.width, image.height );
  }

  var testedRenderers = [ 'canvas', 'svg', 'dom', 'webgl' ];

  // known clipping issues to fix
  var nonDomWebGLTestedRenderers = testedRenderers.filter( function( renderer ) { return renderer !== 'dom' && renderer !== 'webgl'; } );

  /* eslint-disable no-undef */
  // We can only guarantee comparisons for Firefox and Chrome
  if ( !platform.firefox && !platform.chromium ) {
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
    QUnit.test( name, function( assert ) {
      var done = assert.async();
      // set up the scene/display
      var scene = new Node();
      var display = new Display( scene, {
        preserveDrawingBuffer: true
      } );

      function loadImages() {
        // called when both images have been loaded
        function compareSnapshots() {
          var referenceSnapshot = snapshotFromImage( referenceImage );
          var freshSnapshot = snapshotFromImage( freshImage );

          display.domElement.style.position = 'relative'; // don't have it be absolutely positioned
          display.domElement.style.border = '1px solid black'; // border

          // the actual comparison statement
          snapshotEquals( assert, freshSnapshot, referenceSnapshot, threshold, name, display.domElement );

          // tell qunit that we're done? (that's what the old version did, seems potentially wrong but working?)
          done();
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
          assert.ok( false, name + ' reference image failed to load' );
          done();
        };
        freshImage.onerror = function() {
          assert.ok( false, name + ' fresh image failed to load' );
          done();
        };

        referenceImage.src = dataURL;

        display.foreignObjectRasterization( function( url ) {
          if ( !url ) {
            assert.ok( false, name + ' failed to rasterize the display' );
            done();
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
      ( function() {
        var renderer = renderers[ i ];

        pixelTest( name + ' (' + renderer + ')', function( scene, display, asyncCallback ) {
          scene.renderer = renderer;
          setup( scene, display, asyncCallback );
        }, dataURL, threshold, isAsync );
      } )();
    }
  }

  var simpleRectangleDataURL = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAAcElEQVRYR+3YwQoAIQhFUfv/j572NQRiQTOc1ipyn0+kFpe/dnl/ocGqQgieJPhUiyfzX9VcSazBgTCCyZGbwhFEcCRgzVgzVVcgiGDE8uS3ZpiESZgkNwMO1hyvORpBBD938lcl25Lv+62KEcHfE+wTtBwp2K8YwAAAAABJRU5ErkJggg==';
  multipleRendererTest( 'Simple Rectangle',
    function( scene, display ) {
      display.width = 40;
      display.height = 40;
      scene.addChild( new Rectangle( 6, 6, 28, 28, {
        fill: '#000000'
      } ) );
      display.updateDisplay();
    }, simpleRectangleDataURL,
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Shifted Rectangle',
    function( scene, display ) {
      display.width = 40;
      display.height = 40;
      scene.addChild( new Rectangle( 0, 0, 28, 28, {
        fill: '#000000',
        x: 6,
        y: 6
      } ) );
      display.updateDisplay();
    }, simpleRectangleDataURL,
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Delay-shifted Rectangle',
    function( scene, display ) {
      display.width = 40;
      display.height = 40;
      var rect = new Rectangle( 0, 0, 28, 28, {
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
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Color-change Rectangle',
    function( scene, display ) {
      display.width = 40;
      display.height = 40;
      var rect = new Rectangle( 6, 6, 28, 28, {
        fill: 'green'
      } );
      scene.addChild( rect );
      display.updateDisplay();
      rect.fill = 'black';
      display.updateDisplay();
    }, simpleRectangleDataURL,
    DEFAULT_THRESHOLD, testedRenderers
  );

  // try rendering the image itself
  multipleRendererTest( 'Image with PNG data URL',
    function( scene, display, asyncCallback ) {
      var img = document.createElement( 'img' );
      img.onload = function() {
        display.width = 40;
        display.height = 40;
        scene.addChild( new Image( img ) );
        display.updateDisplay();

        asyncCallback();
      };
      img.error = function() {
        asyncCallback();
      };
      img.src = simpleRectangleDataURL;
    }, simpleRectangleDataURL,
    DEFAULT_THRESHOLD, testedRenderers, true // asynchronous
  );

  multipleRendererTest( 'Color change from property',
    function( scene, display ) {
      display.width = 40;
      display.height = 40;
      var colorProperty = new Property( 'red' );
      scene.addChild( new Rectangle( 6, 6, 28, 28, {
        fill: colorProperty
      } ) );
      display.updateDisplay();
      colorProperty.set( 'black' );
      display.updateDisplay();
    }, simpleRectangleDataURL,
    DEFAULT_THRESHOLD, testedRenderers
  );

  /* eslint-enable */

  multipleRendererTest( 'Invisible node with rectangles (paths) above and below',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var shape = Shape.rectangle( 0, 0, 30, 30 );
      scene.addChild( new Path( shape, {
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: -10, y: -10
      } ) );
      scene.addChild( new Path( shape, {
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: 10, y: 10,
        visible: false
      } ) );
      scene.addChild( new Path( shape, {
        fill: '#000',
        stroke: '#f00',
        lineWidth: 2,
        x: 20, y: 20
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAWklEQVRYR+3WsQoAIAhF0ef/f3TZ0NAQWQgR3ajN5HkmTVLxl3J6I5P8xk4rJAACCCCAAAIIIPC5gI+fJhBbwsYqux7gJPXsTx8mvJF6o53aZVYCIIAAAk8KVE6CQBKyrnp4AAAAAElFTkSuQmCC',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Invisible => Visible',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var rect = new Rectangle( 0, 0, 16, 16, { fill: '#f00', visible: false } );
      scene.addChild( rect );
      scene.addChild( new Rectangle( 0, 0, 32, 32, { fill: 'green', visible: false } ) );
      display.updateDisplay();
      rect.visible = true;
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3WMQoAMAgDwPj/R7e+wQpdTrIK4Ratk3TmU0lnPqUAAQIECBAg8F1gfsl3Np+eiY0KChAgQIAAAQIX6VUgIfXDabwAAAAASUVORK5CYII=',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Invisible repaints',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var rect = new Rectangle( 0, 0, 16, 16, { fill: '#f00' } );
      scene.addChild( rect );
      display.updateDisplay();
      rect.visible = false;
      display.updateDisplay();
      rect.x = 16;
      rect.visible = true;
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAQUlEQVRYR+3WMQoAMAgDQP3/o9tCf2AGlxPXQLhFu8I5VW/n0/PoTypAgAABAgQIrAuk5zz+BxQgQIAAAQIEUoELbg4gAWKut4YAAAAASUVORK5CYII=',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Invisible Rectangle',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var rect = new Rectangle( 0, 0, 16, 16, { fill: '#f00', visible: false } );
      scene.addChild( rect );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAALUlEQVRYR+3QQREAAAABQfqXFsNnFTizzXk99+MAAQIECBAgQIAAAQIECBAgMBo/ACHo7lH9AAAAAElFTkSuQmCC',
    DEFAULT_THRESHOLD, testedRenderers
  );

  /*---------------------------------------------------------------------------*
  * Circles
  *----------------------------------------------------------------------------*/

  var redCenteredCircle = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAABLElEQVRYR+2W0XHCMAyGP01QOgllAjpC2aCM0AlKNmADugF0gzJBYRM6gTldzDUcsWXHufNL9JKHONIX6bd/C5VDKtdnApg6kN0BB2/AHHgFXryIT8APcBL4zhF2MoBri+06RUN1FGYtoE8zkgAcbIBPM9v9go1AY31jAgwsfqv7IbCNQUQBfNt/rb8w3i9i47AAdI4quJJQYS5CCYIAXu37ksqdb1cCh75cMYAhwgvxNtIK+SFiALqvlyN14CjtuZEFcAGeRgK4CDzXBPgTmOUCVB9BdRGq6dTbhjov1xpK6UF0ln/XTN8FHkAdsN5R7CFKtFBmRrd+DXREs7jmN+24A6Hj+ErQxBl4H/VC0lWONymFCV3Jek0n2w1HOoLNNMkjMDMNXDABTB24AvQKQiGCr2i0AAAAAElFTkSuQmCC';
  multipleRendererTest( 'Simple Circle',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var circle = new Circle( 10, { fill: 'red', centerX: 16, centerY: 16 } );
      scene.addChild( circle );
      display.updateDisplay();
    }, redCenteredCircle,
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Shifted Repainted Circle',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var circle = new Circle( 10, { fill: 'black' } );
      scene.addChild( circle );
      display.updateDisplay();
      circle.x = 16;
      circle.y = 16;
      circle.fill = 'red';
      display.updateDisplay();
    }, redCenteredCircle,
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Scaled Circle',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var circle = new Circle( 5, { fill: 'red', scale: 2, centerX: 16, centerY: 16 } );
      scene.addChild( circle );
      display.updateDisplay();
    }, redCenteredCircle,
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Radius Change Circle',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var circle = new Circle( 5, { fill: 'red', centerX: 16, centerY: 16 } );
      scene.addChild( circle );
      display.updateDisplay();
      circle.radius = 10;
      display.updateDisplay();
    }, redCenteredCircle,
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Static Circles',
    function( scene, display ) {
      display.width = 64;
      display.height = 64;
      scene.addChild( new Circle( 32, {
        fill: new LinearGradient( 0, -20, 0, 20 ).addColorStop( 0, 'green' ).addColorStop( 1, 'blue' ),
        rotation: Math.PI / 2,
        scale: 0.7,
        center: new Vector2( 32, 32 )
      } ) );
      scene.addChild( new Circle( 10, { fill: 'red', stroke: 'rgba(0,0,255,0.5)', lineWidth: 3 } ) );
      scene.addChild( new Circle( 75, { stroke: 'black', lineWidth: 4 } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,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',
    DEFAULT_THRESHOLD, testedRenderers
  );

  /*---------------------------------------------------------------------------*
  * General tests
  *----------------------------------------------------------------------------*/

  multipleRendererTest( 'Simple Node/Fill/Stroke Opacity',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      scene.addChild( new Rectangle( 0, 0, 32, 16, {
        opacity: 0.7,
        fill: 'rgba(0,0,0,0.7)',
        stroke: 'rgba(0,0,255,0.7)',
        lineWidth: 10
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAWUlEQVRYR2NkYDi6iGEAAeOoA0ZDAHsIWN+mTbo8qopuLo5EOOqA0RAY8BCgTR7AZupoUTwaAowMDP9r6ZfkMG0adcBoCDAOZAIE2T3qgNEQGA2B0RAYDQEAVuAp/TzifcQAAAAASUVORK5CYII=',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Nested Node/Fill/Stroke Opacity',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      var rect = new Rectangle( 0, 0, 32, 16, {
        opacity: 0.7,
        fill: 'rgba(0,0,0,0.7)',
        stroke: 'rgba(0,0,255,0.7)',
        lineWidth: 10
      } );
      rect.addChild( new Circle( 15, {
        opacity: 0.7,
        fill: 'rgba(0,255,0,0.7)',
        stroke: 'rgba(255,0,0,0.7)',
        lineWidth: 5,
        top: 16
      } ) );
      scene.addChild( rect );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAACBklEQVRYR+2US2gTURSGv5tJiFPbmBGLiI9CoWAtWG13irhVXLkoBlc+doJ02267cCUU6cJFrYgg1G5cCAoibqSC1FfBpqIgLVJE2nSSljaTJpPIiZkQYlpjOkk3czdzuHPn/N/895yjYPIhO7iUB+A5UNmB09/qU5eTHeV5NylCD8BzYMcdqE8PVMrqjWLPATXCjbsJjIyJYacIZqfptqbotSyaco0oxTxAJaGfHEg/4WJilq6NeoKo4X2XR31WMEtW4TxLBedoTz3j3Gq9QNTto+fvlwoqK5DTVlpsLR6yBcpZE0TM15xJuu3GXwBFAVuhmWHbvxTOOHvv6V1/wLW4mxCKHJfa5tEOLuA7+xK9O0rw0DwBR0RPku6cZVnfwC7sfVcw5RZEHqA82ak3BAZuYeyN4Zd3fptsZ5RYaI104exnBTNuQFQEcBIPDhG68JzmTSDeKpjbLsSWAJK8bxy9/w6GxLvSZE58YlGzkRkh7flUQbFGaoH5J4AkvT7G7qv32COxsULyWBTTrauoCkDERm5inHyHLnHXF5bC8bwD23ahagDplNErtDZZ+FpWSR2fIVZw4aOCr7XYL99UDSCH+4dp7psgJHHPB34VWtNU8KIhAOLCowj7RezIAonDP1j78xc8bgiAiIxHaJVBVVaMrxQs1gLxX1dQWowyIXumi6I1z4TfMzTM/WJEgl8AAAAASUVORK5CYII=',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Simple clipping',
    function( scene, display ) {
      display.width = 32;
      display.height = 32;
      scene.addChild( new Rectangle( 2, 2, 28, 28, {
        fill: 'black',
        clipArea: Shape.circle( 0, 16, 12.5 ),
        x: 10,
        y: -10
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAzElEQVRYR+2WUQ0CMRBE3zngJOAACeDgcIAEcAAOzgIOkIAEJIADUACZhA9CQm7atOkH29/bdmffTDbXkX6expUVcDbq6JyirxpHwBHYOG/XEnAH+pYC1NuyoRYBCdgB4xSFmgIOwP6vBayBU0sCTUP4AGZT0+t7rRA2X0Rz4NqKgD19DQtuwALQKrZOyQwoeEvgYnV+F5USoMmH1OalLJDn2xTsn4RyCQi3tpx2vZX2X7bkCJDP1t+Ok4UcAc67dk0ICAJBIAgEgSDwAkQfKyGvxnQiAAAAAElFTkSuQmCC',
    1, nonDomWebGLTestedRenderers
  );

  multipleRendererTest( 'Nested clipping and background',
    function( scene, display ) {
      display.width = 64;
      display.height = 40;
      var clipShape1 = Shape.regularPolygon( 6, 27 ).transformed( Matrix3.translation( 32, 32 ) );
      var clipShape2 = Shape.regularPolygon( 9, 32 ).transformed( Matrix3.translation( 32, 0 ) );
      var shape = Shape.circle( 32, 32, 25 );
      var circle = new Path( shape, {
        fill: '#000',
        stroke: '#f00',
        lineWidth: 3
      } );
      circle.clipArea = clipShape1;
      var wrapper = new Node();
      wrapper.addChild( circle );
      wrapper.clipArea = clipShape2;
      scene.addChild( wrapper );
      display.backgroundColor = '#eee';
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAoCAYAAABOzvzpAAADb0lEQVRoQ+2ZMUgbYRTH/zEWY9WkCC6SiHVwKKaiSJaEKg7ikkBxqRqspYMKDtYOQbGWtqJk0DoqUmqlhG5ChFSctCQQRBya0LXViBSE0KS1Ru35yheIWI16393FFLyDg4N733v/97v3/77hNNFolHCNL40KQJ0A1QLqHnCN90Com6B6Cih4CmjDYcr1+5ETiyGRSCAajWJ7ezv5zK4/Wi1+FhQgodNhLy/vH+fl7+9Dl0igaHcXuYKQfKfT6VBaWori4uLk85HBgMPGRhxVVmqUsq0yFhAEKmxvx42lpTPCvADNA1gG8A0QJbwcoAYA9wE40qw5bGqiXx4PkJMjKt9FsBQBcHNggPKmp4/FxAGaBPAawA+RTZ8n8hZATwD0AdCfyLXf1UW/x8ayD0AbCpG+vj4pRMnGTwNJByK+skKC2SwLguwJiNXUUPnGhuYTQA85xlyqh5k93gG4B2giFRVUuLaWPQBfJyaodmRE8xKg5zJHnRfIC4CGAc360BDd7u+XDEHWBOyWlNAzQcDsFTefgtUJ0CutFgU7O1cPYN1up/eBQNaaPwnBabWidmFBEgRJE/A9GKTxlha82duTVJR33C+Lf5yfT8PBIIpMJm49kgC8dTjoqd/PXeyyRuS8H7fZ6JHXy62JG8DnuTlq6OvjLiSnObFrlycn6W5HB5c2bgCdzc3kXV3lKiK2AblxDouFZhcXubRxAVhyu+mB281VQG5TvOs/uFzU5HKJ1igagHBwQPV1dfiytSU6Oa94JeLvGI300edDkdEoSqdoAEycx+NBb2+vEjozlsNkMmF+ZoYqLBblATDVg4ODmJqaylgDUhPr9Xr09PTA5XJxpeCagFRmp9MJn8/HVSiTwa2trRgdHYXBYOAuIwlALBaD3W5HOBzmLqjkAqvVmmzcbDZLTisJAKvGIFRXVyMej0suLnUh8zkb9ba2NqkpjtdJBsAyhEKh5J4QCARkCxGTIOXz7u5uSeOeroYsACcTMhibm5tJW7BndkciETF9iYphPmdfvaysTFS82CDFAJxX0O/3nwHDY5uqqqqkz202m9ieuOIyDiCdGrZ/sAlh1mFTw+7TNmLjzhpXwucXEckKgPMEpWzEgLDGpRxrXJ8fUP8M/VcTwPv1lIhXAaj/BhX8N6jESF51DtUCqgWuuQX+Atsjs6DpctS5AAAAAElFTkSuQmCC',
    2.5, testedRenderers // higher threshold due to antialiasing :(
  );

  multipleRendererTest( 'Ellipses and Elliptical Arcs',
    function( scene, display ) {
      display.width = 200;
      display.height = 50;
      var x = 32;
      scene.addChild( new Path( Shape.ellipse( x, 25, 20, 10 ), {
        fill: '#ff0000',
        stroke: '#000000',
        lineWidth: 3
      } ) );

      x += 64;
      scene.addChild( new Path( new Shape().ellipticalArc( x, 25, 20, 10, Math.PI / 6, Math.PI / 4, Math.PI * 3 / 2, false ), {
        fill: 'rgba(255,0,0,0.5)',
        stroke: '#aa0000',
        lineWidth: 3
      } ) );

      scene.addChild( new Path( new Shape().ellipticalArc( x, 25, 20, 10, Math.PI / 6, Math.PI / 4, Math.PI * 3 / 2, true ), {
        fill: 'rgba(0,0,255,0.5)',
        stroke: '#0000aa',
        lineWidth: 3
      } ) );

      x += 64;
      scene.addChild( new Path( new Shape().moveTo( 0, 0 )
        .ellipticalArc( 0, 0, 20, 10, 0, 0, Math.PI / 3, false )
        .ellipticalArc( 0, 0, 20, 10, 0, Math.PI / 3 + Math.PI, Math.PI, true )
        .close(), {
        x: x,
        y: 25.5,
        fill: '#ff0000',
        stroke: '#000000',
        lineWidth: 3
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,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',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Circles and Arcs',
    function( scene, display ) {
      display.width = 200;
      display.height = 50;
      var x = 32;
      scene.addChild( new Path( Shape.circle( x, 25, 20 ), {
        fill: '#ff0000',
        stroke: '#000000',
        lineWidth: 3
      } ) );

      x += 64;
      scene.addChild( new Path( Shape.arc( x, 25, 20, Math.PI / 4, Math.PI * 3 / 2, false ), {
        fill: 'rgba(255,0,0,0.5)',
        stroke: '#aa0000',
        lineWidth: 3
      } ) );

      scene.addChild( new Path( Shape.arc( x, 25, 20, Math.PI / 4, Math.PI * 3 / 2, true ), {
        fill: 'rgba(0,0,255,0.5)',
        stroke: '#0000aa',
        lineWidth: 3
      } ) );

      x += 64;
      scene.addChild( new Path( new Shape().moveTo( 0, 0 )
        .arc( 0, 0, 20, 0, Math.PI / 3, false )
        .arc( 0, 0, 20, Math.PI / 3 + Math.PI, Math.PI, true )
        .close(), {
        x: x,
        y: 25.5,
        fill: '#ff0000',
        stroke: '#000000',
        lineWidth: 3
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,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',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'DAG support',
    function( scene, display ) {
      display.width = 250;
      display.height = 120;
      var square = new Path( Shape.rectangle( 0, 0, 50, 50 ), {
        x: 0.5,
        y: 0.5,
        fill: 'rgba(0,0,0,0.5)',
        stroke: '#000000',
        lineWidth: 3
      } );

      var arrow = new Path( Shape.regularPolygon( 3, 10 ), {
        fill: 'rgba(0,255,0,1)',
        stroke: '#000000'
      } );

      var combined = new Node();
      combined.addChild( square );
      var leftArrow = new Node( { x: 15, y: 25 } );
      leftArrow.addChild( arrow );
      combined.addChild( leftArrow );
      var rightArrow = new Node( { x: 35, y: 25 } );
      rightArrow.addChild( arrow );
      combined.addChild( rightArrow );

      var a = new Node( { x: 20, y: 20 } );
      a.addChild( combined );
      var b = new Node( { x: 120, y: 30, rotation: Math.PI / 4 } );
      b.addChild( combined );
      var c = new Node( { x: 220, y: 40, rotation: Math.PI / 2 } );
      c.addChild( combined );
      scene.addChild( a );
      scene.addChild( b );
      scene.addChild( c );
      display.updateDisplay();
    }, 'data:image/png;base64,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',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Overlapping/nested transforms',
    function( scene, display ) {
      display.width = 64;
      display.height = 44;
      var container = new Node( {
        x: 20,
        rotation: Math.PI / 12,
        scale: 0.5
      } );
      scene.addChild( container );

      container.addChild( new Path( Shape.rectangle( 0, 0, 44, 44 ), {
        x: 10.5,
        y: 10.5,
        rotation: Math.PI / 4,
        fill: '#ff0000',
        stroke: '#000000',
        lineWidth: 3
      } ) );

      container.addChild( new Path( Shape.rectangle( 0, 0, 44, 44 ), {
        x: 30.5,
        y: 10.5,
        fill: '#ff0000',
        stroke: '#000000',
        lineWidth: 10
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAsCAYAAADVX77/AAAEt0lEQVRoQ+2Zy2+NQRjGfz09grjVSuJSbMV9J4La2FhgLVH+AOq2d4mtuPwDVGKt1hZFWCMklm4RS6TsTg95vs5X4+vMNzPnmzZN2jdpOD1zeeeZ53kv0z4WuPUt8POzCMAiAxY4ArMtgePAU+DHfMV5tgDYDdwChoBR4LQHgNvAIQPSGPBsroHKDcAAcAU4vwq6e6FlTrQHeO043CtAYNkmxpRguOZkxSgnACMtuN6F1SPAVePmIHQn4DlwuOK5wPoeOI2kYwPyMevpIUsaHGrBaBcGxeX7wBbLS30+M/VZ/+hjaZLFvcQDCQCxQ6CIXI1jSxMG6Jw6wNAm6D6AlgTvsp0w+R6+dWCH5bTAGE4EoDpcErEBSV6uFwBE3YLlK2HyEvSXdPftrusy/L9mqeNDhSzJzjsm2GBExY9UAIZbcFc619UphAuNGBPfH8IvwwJNEQD/mSKiwNLP45hF68dIHjYgzvgRC8BQG+7KeelcB6+G7pC/2n07dH4XOBQZQWly2spcaP+uBEOneBPaoP77O8pMriEhAKRzpbXTG6BzA9q+hB7jn6QiDQAvgf32HG1SJ6UyHZQp4VPMhv/GSIGaOsN8ABQ6b8PlpbDsMrQFXyzdfb7pEGLB16ns02+PcxUEdWcUo2yG/KwHxHvRri+G23CjAxulc92KndbSgJ85Wry/WPn1mgz5TJqSr47YoXTpS1Az6gCd9cMg/BmFPu+sBihYMphe5ZiJVg2WLaaq8XAAcMGELefyLgYU+Xk2cpQ8EKjVgl+scEaoREQ8PPeV4cXqrjkDLfh8EFaNJzoQGq4YsNYxKAfYVq1h76DQUBu6fMFBF3JLAOSUgdLZiQoAm4EcBb5LWoFO1MuA4guxYCNs+FT8N48JVSVk2xRo7Qah1508PK/2HzOWr6sDdPnjofyc4rDLSTUTTWoL7e+TFrA1RLBQIfR0NRwQC5rWAKK5vKma+uGma7ukBahWCmbwEABFWsxBU6stnsZgl+eVJIVVGisG6dmpYt7y1x4XAkBjFV+upFZqVW9cTqqlVF/R1Dw8V7wVOWotBoAsadHlZI4s45OWybjBB5MYAEqW3Xtkqq0QqtXvVaYqAFbtT+pCjvEuaZnmMaphjQWANrxdD9ve9BAQRXPVo7YpuJRPwk2CoKf8tR9eGkugXKDntHgEeFLjhhbWj3qCqGuz1lJl6eC5t/2tuhHNADNxbCUcfQvtYH4xjikUh57MbKfEBhuQun180kp57E0FYEsb3p2EFaHqTeF3BLqfG1aSAsAGxJaLS1qmIZQyoiwVAC1apEVfBNetSO9qTgRWB84aT+SUzqL037NJIiUg6iIdf0qqbX+bSkDzB3SwfbDueXHGKZMOtbOY0YKJLpzzlPm6xBIMnUX9UE6rbX9zAKA1ij9qlHW8Qu5NmJyYeubSR7EzmIONM2K5DYgeiHq1qPLXXrwXCRTzl8CL5bBP12l0LjYKmKbdrVheAqLH4hSr+0Osc52eATBSHG/Bly6c8r26pnjvGGsnhZj4EWx/c0mgXEdOOZ+bGx7cN91OCq74EWx/cwMwS+eMXtZOCiVboidrYBMJJG00XwcvAjBfb2au/FpkwFwhPV/3+QuuPfkt5gxIKAAAAABJRU5ErkJggg==',
    DEFAULT_THRESHOLD, testedRenderers
  );

  // 16x16 with black circle in the middle
  var patternUrl = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAo0lEQVQ4T7WT4Q2CMBCFPyZAN2EDcAOcQJ1MmABHkA0YBTcwL6FELhYvFpr0B+n13X2vj4zElSXeJyZwBS5AMTUYgAZobUMrcAA6oIpM9gTOwBjOrYAKyh9Yqjl9E9DYd6cntwlp4YGne9DvA+Yngrhy5wSqPar2X4EXIMO3RUg2URN5jJwNtAj6FtdjJQu6XK8FKTyCcLRtlBXnxdrtZ3LGAd6BXBwR28m13gAAAABJRU5ErkJggg==';

  multipleRendererTest( 'Patterns',
    function( scene, display, asyncCallback ) {
      display.width = 64;
      display.height = 64;
      var img = document.createElement( 'img' );
      img.onload = function() {
        var pattern = new Pattern( img );
        scene.addChild( new Path( Shape.regularPolygon( 6, 22 ), {
          x: 32,
          y: 32,
          rotation: Math.PI / 4,
          fill: pattern,
          stroke: '#000000',
          lineWidth: 3
        } ) );
        display.updateDisplay();
        asyncCallback();
      };
      img.src = patternUrl;
    }, 'data:image/png;base64,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',
    DEFAULT_THRESHOLD, testedRenderers, true
  );

  multipleRendererTest( 'Transformed Patterns',
    function( scene, display, asyncCallback ) {
      display.width = 64;
      display.height = 64;
      var img = document.createElement( 'img' );
      img.onload = function() {
        var pattern = new Pattern( img ).setTransformMatrix( Matrix3.rowMajor( 0.4, 0.2, 1.3, -0.2, 0.2, 7.2, 0, 0, 1 ) );
        scene.addChild( new Path( Shape.regularPolygon( 6, 22 ), {
          x: 32,
          y: 32,
          rotation: Math.PI / 4,
          fill: pattern,
          stroke: '#000000',
          lineWidth: 3
        } ) );
        display.updateDisplay();
        asyncCallback();
      };
      img.src = patternUrl;
    }, 'data:image/png;base64,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',
    2.8, testedRenderers, true // larger threshold due to antialiasing
  );

  multipleRendererTest( 'Dashes',
    function( scene, display, asyncCallback ) {
      display.width = 64;
      display.height = 64;
      scene.addChild( new Path( Shape.rectangle( 10, 10, 44, 44 ), {
        stroke: '#000',
        lineWidth: 3,
        lineDash: [ 7, 2, 3, 2 ],
        x: 0.5,
        y: 0.5
      } ) );
      scene.addChild( new Rectangle( 5, 5, 54, 54, {
        stroke: '#000',
        lineWidth: 3,
        lineDash: [ 7, 2, 3, 2 ],
        x: 0.5,
        y: 0.5
      } ) );
      scene.addChild( new Path( Shape.regularPolygon( 6, 15 ), {
        stroke: '#000',
        lineDash: [ 5, 5 ],
        lineDashOffset: 5,
        x: 32.5,
        y: 32.5
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAADd0lEQVR4Xu2bva8OQRTGfzeIAoUKCaLjX1AICjpREUpKofSRKJCQEC1Kah+RaGlufEStEqIQJCIhChQSQo7svjZ75919Zs5m3919d5JN7n33nJlznjnnmTO7swvMeVuYc/8ZARgjYCkCf2pAuQCcz65zmWzot3I3XdKb2BZKgRGAMQL+IzAkkszT1byzdPzX6lJgSAAUU3viVwwARnxFFIuZ0iWCK9uZ+zgCUJixyggwBPOW/20zvyu7Qhw5qAgIORgMnw4XUSF75RQIseVcASCj17MICKW2vAz2LQKCzoYmLLQMhpT7BkDIV7kQGgIJyjwWioCQshxSHeEFmcdiKsFU33YDv4HHhQ6WAb9KHdr/K0q/ma45sxg5eAgAFwkG80cwah3wAlgvyE4T+QxsBb5E9CFHrBoBqSR4F7gN3Iswvix6CNgPHHb0YaoyCcr5U2PQzmzbaSW0tz0FTgPPxI5cJCjnT40x94FTwBvR6CoxS4GLwAGxL3kS1RQQx+2MmAuAabvB3LvJ05TOuLvUEBcAQyiEXADIBDLUCJDRCwBwA7iTULjEYGmrykHgWIWSTOQqCSp1wB7gJLA3xptE2YfAVeDRFH1XIZS6G3wF7ANeJzoVo2bL4gNgW4SSXAilkODxrFw9EWGQV/Qa8BK4HuhI5rGmdoM24CyWx7PApQAAMo+pHOCdrbb1GyfB1N1g247n47lIUA6fWXmXOK5Mgh4ANgNfgW+JRipqa4C1wLsKYRcJyvkTMGAj8BzYpHiSKPMe2A58iCyE5BcjStFTZfvlLAquJDpYpWbPBGz2z9T0LUex+lg8lgS/A/Y47EeDIKwCPgGrhT5dAKQUQmWdI8AO4KhgrCpyE3gC3BIUXADIBFJjyBbgI/BTMLhOZCWwAXhbJ5jddwEgK4vGzEJMJnK1EvQSY9sguAqh1N1glZP2IqTc7GVJualyKYDKhVATJFjuo/wWyO4vDwykytUBIPNYU7vBOoPavi/zmMoBbTvgHa9xEowthLwOePVdJCiHj9fKlvVlEhwCAC4SlPOn5RmMGU6exHkiQXk7HHtStMunRC1qos8K1xVC5fuDA6COQAYPQN82Pio5ug5JqYP0Tq7qkZi9hbVzPqHW9bzPbS7bacft7Kr8ZCZX7usXIsUJm/Y5X+UHEyMAvUtkh8FD+iosCYa5B+AvQoqBUDmT0twAAAAASUVORK5CYII=',
    2.5, testedRenderers
  );

  multipleRendererTest( 'Radial Gradients',
    function( scene, display, asyncCallback ) {
      display.width = 192;
      display.height = 64;
      var fillGradient = new RadialGradient( 22, 0, 0, 22, 0, 44 );
      fillGradient.addColorStop( 0.1, '#ff0000' );
      fillGradient.addColorStop( 0.5, '#00ff00' );
      fillGradient.addColorStop( 0.6, 'rgba(0,255,0,0.3)' );
      fillGradient.addColorStop( 0.9, '#000000' );

      var mainGradient = new RadialGradient( 32, 20, 10, 32, 32, 32 );
      mainGradient.addColorStop( 0, '#8ED6FF' );
      mainGradient.addColorStop( 0.5, '#004CB3' );
      mainGradient.addColorStop( 0.6, '#bbbbbb' );
      mainGradient.addColorStop( 1, '#ffffff' );

      var transformedGradient = new RadialGradient( 0, 0, 0, 0, 0, 64 );
      transformedGradient.addColorStop( 0.3, '#8ED6FF' );
      transformedGradient.addColorStop( 1, '#004CB3' );
      transformedGradient.setTransformMatrix( Matrix3.translation( 32, 32 ).timesMatrix( Matrix3.rotation2( Math.PI / 4 ).timesMatrix( Matrix3.scaling( 1, 0.25 ) ) ) );

      scene.addChild( new Path( Shape.rectangle( 0, 0, 64, 64 ), {
        fill: mainGradient
      } ) );
      scene.addChild( new Path( Shape.regularPolygon( 6, 22 ), {
        x: 32 + 64,
        y: 32,
        rotation: Math.PI / 4,
        fill: fillGradient,
        stroke: '#000000',
        lineWidth: 3
      } ) );
      scene.addChild( new Path( Shape.rectangle( 0, 0, 64, 64 ), {
        x: 128,
        fill: transformedGradient
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,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',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Linear Gradients',
    function( scene, display, asyncCallback ) {
      display.width = 128;
      display.height = 64;
      var fillGradient = new LinearGradient( -22, 0, 22, 0 );
      fillGradient.addColorStop( 0.1, '#ff0000' );
      fillGradient.addColorStop( 0.5, '#00ff00' );
      fillGradient.addColorStop( 0.6, 'rgba(0,255,0,0.3)' );
      fillGradient.addColorStop( 0.9, '#000000' );

      var strokeGradient = new LinearGradient( 0, -15, 0, 15 );
      strokeGradient.addColorStop( 0, '#ff0000' );
      strokeGradient.addColorStop( 0.5, '#00ff00' );
      strokeGradient.addColorStop( 1, '#0000ff' );

      scene.addChild( new Path( Shape.regularPolygon( 6, 22 ), {
        x: 32,
        y: 32,
        rotation: Math.PI / 4,
        scale: 0.5,
        fill: fillGradient,
        stroke: '#000000',
        lineWidth: 3
      } ) );
      scene.addChild( new Path( Shape.regularPolygon( 6, 22 ), {
        x: 32 + 64,
        y: 32,
        fill: fillGradient,
        stroke: strokeGradient,
        lineWidth: 3
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,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',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Cubic B\u00E9zier',
    function( scene, display, asyncCallback ) {
      display.width = 64;
      display.height = 64;
      scene.addChild( new Path( new Shape().moveTo( -20, -20 ).cubicCurveTo( -20, 0, 0, 0, 20, 20 ).lineTo( 20, -20 ).close(), {
        x: 32,
        y: 32,
        fill: '#ff0000',
        stroke: '#000000',
        lineWidth: 3
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAACqUlEQVR4Xu2a61UbMRCFLxWEDqCD0AF0AB2QDqADoALoADqADiAVJB2QDkIJnM/HI9YvJBt2pJU0v3ywWGu+vXdm9rGnxmOv8fzVAXQFNE6gW6BxAfQi2C3QLZBO4ETScfryrCt/S3pJ2cE2FriWdJVy0ALW3Ehiv9HoAKKIPhYEBeAD/FBSoHd0P49RFEDOz/zAqaSnkrKf652sXQCggKQK4wgJeY4NYF/Sf3IKHxwTjP2UBwD28E/SAR/+SDqK7crxey8AWJ8SoFtJl44Jxn7KC8AvSfdshrOPCkoJLwDk+ybpBx9oCaW0Q08AD5LOARD6YgEy8ARwKOnVcsYP+CJ3eAIg1zAV0hKxQu6O4A2AvP9aS0QSFET+mCu8AVgjYCCcFcTcXSEHAPI+k/RoZx0I2CGHEnIBIPcwG5gSKIzeNSEngBUIKABZeM4IuQGYHZgRZjWBYFRmZPaIEgCsFEZPS5QCgJxxAEqYXTRZjH1DsSQAQwewr2AJCiOWGKM2lAgAEMxIqGHhdjq1gdvL39kuSwWwUQ0kT7tkkPiOKB2AqeFuuTZgB0Agla/EFABYfpx0QMxurVnVNFvsCmFKACxn9nwxTJgiuesUOTUAljc5UyR/LrdMyGxTJKcKYGORpCYwTqdeU0wdwMaWmTpA1QDg0wEKNXzWKWoCsFYNsbmhNgBDNSxcUG66wqwVACCogzyRCnMDwxOWGHaJmgHY3LBwhbncJWoHsNYSw7rQCgBA4AAsES6zmR55bD32+wG7julj/N9KXcASQJjHKK/IjJHIV46JA3gusTBGtwRgbXFsDYApKDyxbhUAeQ8hNFED1tUPg9AsAFMCDYHRIBrbvCobPVhBC2iTPL6PRq0Aoonbgg4gGVWlC7sCKj2xyWl1BSSjqnRh8wp4Bz/QokHSunQ4AAAAAElFTkSuQmCC',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Quadratic B\u00E9zier',
    function( scene, display, asyncCallback ) {
      display.width = 64;
      display.height = 64;
      scene.addChild( new Path( new Shape().moveTo( -20, -20 ).quadraticCurveTo( 20, -20, 20, 20 ).close(), {
        x: 32,
        y: 32,
        fill: '#ff0000',
        stroke: '#000000',
        lineWidth: 3
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAACuklEQVR4Xu3a4XHVMAzAcXUD2IBOQLtBmYAyAe0GdAO6AUxQukHZADagExQ2gA24/11y5yhJK9tK4sT2t97L5Z5+lWQ5eSdS+TqpPH5pAC0DKhdoJVB5ArQm2EqglcDLAmci8n7msr8i8iv47OfLtyvrCmsJfBORjxFfHZTfHc4PEfnT/R1xi3UutQLwbWIRdASAPIgIIGQK2bP5igEYIbwSkbfp+Q8GqN+3VIgFGCFcicidikDnP3//m4+SzADi6xZZkQJgQtDxgkDu82+f6ZSUxJe1IVIBkhB6lL4ZEC3dUS0gPncQi1dHDkAWQh8ZWUG0E1nBR9dL7x65AC4I3IRo6ScqI8gGEKicRZYHgBsCNyIbbsehgkCjdF9eAK4INMzLcTYAAITr8gRwRSD3L0TkcRiuO4I3gCsCN6Mv3A8RXMthCYA1ED54NcalANwROJIG5UCFnHpMjksCuCIQMQjBNsnO+S63Iy4N4IrA7nA+jPimG5+THdYAcEVQc0J2KawF4IrwZlgKzE24JK01AdwQmIvZBrqVlQVrA7ghMCQFB6jkLNgCwAVBbQGcsNkWo9dWAC4IqhckDUdbAmQj8ECFfbBbPFvkDBW1tgbIQpjI+9ex02EJAFkIqhlGl0EpAMkIqgx4svwppgZKAkhCUONx9G5QGkASAi9ogvcObIdAmFaJANEIqg9wQmRMMK1SAaIQ1AEpaiosGcCMoM4GUfNA6QAmBNUIJx4bzFfDHgBMCCoQc1zmC00dZdmLBr9P0G+lawB4NhNSd4I9ZUCfX5OZUBPAZCY8DR+QmGeBPWbAZCYwDQY/OjK/PdozwCgTgh5sHob2DjCHUBXAFEJ1ABqhSoAQoVqAHoHnAaa3RUdoglMDOC+Swx9xzw7pRwUwn0oagJnqoBe2DDjoP9YcVssAM9VBL6w+A/4DQEu9QTwwzr8AAAAASUVORK5CYII=',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Opacity and Blending',
    function( scene, display, asyncCallback ) {
      display.width = 64;
      display.height = 64;
      scene.addChild( new Rectangle( 0, 12, 64, 20, { fill: '#000' } ) );
      var circle = Shape.circle( 0, 0, 30 );
      scene.addChild( new Node( {
        opacity: 0.5,
        children: [
          new Path( circle, { x: 12, y: 22, fill: '#f00' } ),
          new Path( circle, { x: 52, y: 22, fill: '#00f' } )
        ]
      } ) );
      display.updateDisplay();
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAACnklEQVR4Xu2bsVIUQRCGvzGRSLGowhAyjTwziDyMNBLMyOQJxDegeAPwCeQJECKM8Igw0iPSyLtQqiw8IkxcqvfcOj2Oql3d7pmp2avabGqm++u/e2Z2+1wGG3j4/WCaNoec8NDD6qMlnQ8AoTgvGLwAaPHRe+QLDZgDWGeb17z0Kvs/FzcF8JYVnrMbjPOmKSB5P8dXzplOE8Ayu+yzEpTzZgp4zxKPOQzOeTMAst8fsZQmgJCjb6KAZfbY51mQ0VcHIJX/DmfBOp8D2IBMy8JjFnnHE63pa5lXFcAb1ugzV4uhWpOoAtj0c9GsxEoNQI95dnhRyRgfg9UAyPbXoe3Dp0prqgGQi0+XViVjfAxWAxBDAVTdBrdYZxDYzW+SwtQUEMMOoKqABkAEZ4BGAZp3gS1eMeC2j52t0ppqRTD5bTD5g1DyR+HkL0NSiWI4C6gVQQEQQyFUBRDFKzHNz+NRvBTVBCBpkPRrcQGQ/IcRgdCmwxGPKh1RrQab9AeErAITACHXAjMAwwaJPufcslJ3qXXMAIg1SbfIFOFIukmqgNCiywkPSklUe5BpChTODBslO0FA8AJAQIQCwRsAgfCdmdNZTvnFjVltqV83v08AHxwcDA3LngILPiD4APBTdkQHn/92OLsPeSPhTUsQ1gC6wIGDi8lOZlO/IdyzgmAFoC8XQwe9co5l85A3Fqr31wiAVUCL+BfguLzj43hyEIua9rm8BIFIT/66IXn4v9Ql2pLfn66XejkdjEblqVG7feAucgBXuIOQl0dau+URA+6OjRvk2/no6f17pCsDGbdPbBz/DveNYa0pbOyBu5KCEwFUNSfm8Q2AmKNXh+2NAuqgGPMcjQJijl4dtjcKqINizHM0Cog5enXYnrwCLgHr2NK2FZwcxgAAAABJRU5ErkJggg==',
    DEFAULT_THRESHOLD, testedRenderers
  );

  multipleRendererTest( 'Image shifted',
    function( scene, display, asyncCallback ) {
      var img = document.createElement( 'img' );
      img.onload = function() {
        display.width = 40;
        display.height = 40;
        var image = new Image( img );
        scene.addChild( image );
        display.updateDisplay();
        image.x = 10;
        image.y = 5;
        display.updateDisplay();

        asyncCallback();
      };
      img.error = function() {
        asyncCallback();
      };
      img.src = simpleRectangleDataURL;
    }, 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAoCAYAAACM/rhtAAAAb0lEQVRYR+3VQQrAIAwEQP3/o1sKvVYWtwcP4zmBOMZkjsPPPLy+ocD2hQgSbAXafD1IMBC4gpjtkD96UIHb/G8iQYKNgF/c6D25BAkGAgZ1gLQMIUiwETCoGz2rrtUjSDAUsOpCqM+w4wXbCy7zb6TwHA3a1+y0AAAAAElFTkSuQmCC',
    DEFAULT_THRESHOLD, testedRenderers, true // asynchronous
  );

  multipleRendererTest( 'Image changed after display',
    function( scene, display, asyncCallback ) {
      var img1 = document.createElement( 'img' );
      img1.onload = function() {
        display.width = 32;
        display.height = 32;
        var image = new Image( img1 );
        scene.addChild( image );
        display.updateDisplay();

        var img2 = document.createElement( 'img' );
        img2.onload = function() {
          image.image = img2;
          display.updateDisplay();

          asyncCallback();
        };
        img2.error = function() {
          asyncCallback();
        };
        img2.src = redCenteredCircle;
      };
      img1.error = function() {
        asyncCallback();
      };
      img1.src = simpleRectangleDataURL;
    }, redCenteredCircle,
    DEFAULT_THRESHOLD, testedRenderers, true // asynchronous
  );

  multipleRendererTest( 'External image displayed before load, then after load',
    function( scene, display, asyncCallback ) {
      display.width = 64;
      display.height = 64;

      var img = document.createElement( 'img' );
      var image = new Image( img );
      scene.addChild( image );
      display.updateDisplay();

      // add handler after creating the image
      img.onload = function() {
        // a bit of extra time afterwards
        setTimeout( function() {
          display.updateDisplay();

          asyncCallback();
        }, 200 );
      };
      img.error = function() {
        asyncCallback();
      };
      img.src = '../scenery/examples/example-image-1.png';
    }, 'data:image/png;base64,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',
    2, [ 'canvas', 'webgl' ], true // asynchronous, don't test SVG/DOM, since foreignobject doesn't work with external images
  );
} );