// Copyright 2023, University of Colorado Boulder

/**
 * Represents an instruction to execute part of a RenderProgram based on an execution stack
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { ClippableFace, scenery } from '../../../imports.js';
import RenderExecutionStack from './RenderExecutionStack.js';
import Vector4 from '../../../../../dot/js/Vector4.js';
import Vector2 from '../../../../../dot/js/Vector2.js';

export default abstract class RenderInstruction {
  public abstract execute(
    stack: RenderExecutionStack,
    face: ClippableFace | null, // if null AND we have a need set for a face, it is fully covered
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void;
}

const scratchVector = new Vector4( 0, 0, 0, 0 );

export class RenderInstructionPush extends RenderInstruction {
  public constructor(
    public vector: Vector4
  ) {
    super();
  }

  public execute(
    stack: RenderExecutionStack,
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void {
    stack.push( this.vector );
  }
}

export class RenderInstructionMultiplyScalar extends RenderInstruction {
  public constructor(
    public factor: number
  ) {
    super();
  }

  public execute(
    stack: RenderExecutionStack,
    face: ClippableFace | null,
    area: number,
    centroid: Vector2,
    minX: number,
    minY: number,
    maxX: number,
    maxY: number
  ): void {
    stack.readTop( scratchVector );
    scratchVector.multiplyScalar( this.factor );
    stack.writeTop( scratchVector );
  }
}

scenery.register( 'RenderInstruction', RenderInstruction );
