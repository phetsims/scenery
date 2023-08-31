// Copyright 2023, University of Colorado Boulder

/**
 * RenderProgram that provides splitting based on depth, into a RenderStack
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { scenery } from '../../../imports.js';
import Matrix4 from '../../../../../dot/js/Matrix4.js';

export default class RenderDepthSort {
  public static getProjectionMatrix( near: number, far: number, minX: number, minY: number, maxX: number, maxY: number ): Matrix4 {

    const minZ = near;
    const maxZ = far;

    const diffX = maxX - minX;
    const diffY = maxY - minY;
    const diffZ = maxZ - minZ;

    return new Matrix4(
      2 * minZ / diffX, 0, -( maxX + minX ) / diffX, 0,
      0, 2 * minZ / diffY, -( maxY + minY ) / diffY, 0,
      0, 0, maxZ / diffZ, -minZ * maxZ / diffZ,
      0, 0, 1, 0
    );
  }
}

scenery.register( 'RenderDepthSort', RenderDepthSort );
