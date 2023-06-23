// Copyright 2023, University of Colorado Boulder

import { scenery } from '../../imports.js';

/**
 * A pool of GPU buffers that can be reused.
 *
 * TODO: can we reuse some buffers AFTER we ensure the synchronization is clear?
 * TODO: // device.queue.onSubmittedWorkDone().then( () => {} ); and clear?
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

// If we're applying labels to buffers, we'll need to create a new buffer for each allocation.
const APPLY_LABELS = false;
const AGE_TO_FREE = 2;

// Holds a bunch of buffers with COPY_SRC|COPY_DST|STORAGE that we can reuse (but with the ability to aggressively
// control memory usage). With changing scene/window size, we might have buffers that we need to permanently toss.
//
export default class BufferPool {

  private generation = 0;

  // This is definitely unoptimized
  private freeBuffers: BufferEntry[] = [];

  public constructor( public device: GPUDevice ) {}

  private createBuffer( size: number, label: string ): GPUBuffer {
    return this.device.createBuffer( {
      size: Math.max( size, 16 ), // Min of 16 bytes used, copying vello buffer requirements
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,

      // Conditionally apply the label
      // eslint-disable-next-line no-object-spread-on-non-literals
      ...( APPLY_LABELS ? { label: label } : {} )
    } );
  }

  public getBuffer( size: number, label: string ): GPUBuffer {
    if ( !APPLY_LABELS ) {
      for ( let i = 0; i < this.freeBuffers.length; i++ ) {
        const entry = this.freeBuffers[ i ];
        if ( entry.size >= size ) {
          this.freeBuffers.splice( i, 1 );
          return entry.buffer;
        }
      }
    }
    return this.createBuffer( size, label );
  }

  public freeBuffer( buffer: GPUBuffer ): void {
    this.freeBuffers.push( new BufferEntry( buffer, buffer.size, this.generation ) );
  }

  public nextGeneration(): void {
    this.generation++;

    // Clear out unused buffers
    for ( let i = 0; i < this.freeBuffers.length; i++ ) {
      const entry = this.freeBuffers[ i ];
      if ( this.generation - entry.generation > AGE_TO_FREE ) {
        entry.buffer.destroy();
        this.freeBuffers.splice( i, 1 );
        i--;
      }
    }
  }
}

scenery.register( 'BufferPool', BufferPool );

class BufferEntry {
  // TODO: pool these
  public constructor(
    public readonly buffer: GPUBuffer,
    public readonly size: number,
    public generation: number
  ) {}
}