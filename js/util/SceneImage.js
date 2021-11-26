// Copyright 2013-2021, University of Colorado Boulder

/*
 * An HTMLImageElement that is backed by a scene. Call update() on this SceneImage to update the image from the scene.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../imports.js';

class SceneImage {
  /**
   * NOTE: ideally the scene shouldn't use SVG, since rendering that to a canvas takes a callback (and usually requires canvg)
   *
   * @param {Node} scene
   */
  constructor( scene ) {
    this.scene = scene;

    // we write the scene to a canvas, get its data URL, and pass that to the image.
    this.canvas = document.createElement( 'canvas' );
    this.context = this.canvas.getContext( '2d' );

    this.img = document.createElement( 'img' );
    this.update();
  }

  /**
   * NOTE: calling this before the previous update() completes may cause the previous onComplete to not be executed
   * @public
   *
   * @param {function} onComplete
   */
  update( onComplete ) {
    this.scene.updateScene();

    this.canvas.width = this.scene.getSceneWidth();
    this.canvas.height = this.scene.getSceneHeight();

    this.scene.renderToCanvas( this.canvas, this.context, () => {
      const url = this.toDataURL();

      this.img.onload = () => {
        onComplete();
        delete this.img.onload;
      };
      this.img.src = url;
    } );
  }
}

scenery.register( 'SceneImage', SceneImage );
export default SceneImage;