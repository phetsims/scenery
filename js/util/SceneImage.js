// Copyright 2013-2015, University of Colorado Boulder


/*
 * An HTMLImageElement that is backed by a scene. Call update() on this SceneImage to update the image from the scene.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  // NOTE: ideally the scene shouldn't use SVG, since rendering that to a canvas takes a callback (and usually requires canvg)
  function SceneImage( scene ) {
    this.scene = scene;

    // we write the scene to a canvas, get its data URL, and pass that to the image.
    this.canvas = document.createElement( 'canvas' );
    this.context = this.canvas.getContext( '2d' );

    this.img = document.createElement( 'img' );
    this.update();
  }

  scenery.register( 'SceneImage', SceneImage );

  inherit( Object, SceneImage, {
    // NOTE: calling this before the previous update() completes may cause the previous onComplete to not be executed
    update: function( onComplete ) {
      var self = this;

      this.scene.updateScene();

      this.canvas.width = this.scene.getSceneWidth();
      this.canvas.height = this.scene.getSceneHeight();

      this.scene.renderToCanvas( this.canvas, this.context, function() {
        var url = self.toDataURL();

        self.img.onload = function() {
          onComplete();
          delete self.img.onload;
        };
        self.img.src = url;
      } );
    }
  } );

  return SceneImage;
} );
